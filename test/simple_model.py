import os
import logging
from os.path import exists
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from llmplus.io.batch import Batch, collate_batch
from llmplus.models.make_model import make_model
from llmplus.utils.label_smoothing import LabelSmoothing
from llmplus.utils.rate import rate
from llmplus.utils.run_epoch import run_epoch, TrainState
from llmplus.utils.helpers import DummyOptimizer
from llmplus.utils.helpers import DummyScheduler
from llmplus.io.tokenizer import load_tokenizer, tokenizing
from llmplus.io.data_read import TextReader

logger = logging.getLogger(__name__)

# def data_gen(V, _batch_size, nbatches):
#     "Generate random data for a src-tgt copy task."
#     for i in range(nbatches):
#         data = torch.randint(1, V, size=(_batch_size, 10))
#         data[:, -1] = 1
#         src = data.requires_grad_(False).clone().detach()
#         tgt = torch.flip(data, dims=[-1])
#         tgt = tgt.requires_grad_(False).clone().detach()
#         yield Batch(src, tgt, 0)
#
#
# def example_simple_model():
#     V = 11
#     criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
#     model = make_model(V, V, N=2)
#
#     optimizer = torch.optim.Adam(
#         model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9
#     )
#     lr_scheduler = LambdaLR(
#         optimizer=optimizer,
#         lr_lambda=lambda step: rate(
#             step, model_size=model.src_embed[0].d_model, factor=1.0, warmup=400
#         ),
#     )
#
#     _batch_size = 80
#     print(_batch_size)
#     for epoch in range(10):
#         print("Epoch NO: {}".format(epoch))
#         model.train()
#         run_epoch(
#             data_gen(V, _batch_size, 20),
#             model,
#             SimpleLossCompute(model.generator, criterion),
#             optimizer,
#             lr_scheduler,
#             mode="train",
#         )
#         model.eval()
#         run_epoch(
#             data_gen(V, _batch_size, 5),
#             model,
#             SimpleLossCompute(model.generator, criterion),
#             DummyOptimizer(),
#             DummyScheduler(),
#             mode="eval",
#         )[0]
#
#     model.eval()
#     src = torch.LongTensor(
#         [
#             [0, 1, 2, 3, 4, 5, 6, 7, 8, 1]
#         ]
#     )
#     max_len = src.shape[1]
#     src_mask = torch.ones(1, 1, max_len)
#     print(greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=0))
#

class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = (
                self.criterion(
                    x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
                )
                / norm
        )
        return sloss.data * norm, sloss


def yield_tokens(data_iter, tokenizer, index):
    for from_to_tuple in data_iter:
        # yield str(from_to_tuple[index]).strip().split()
        yield tokenizer(from_to_tuple[index])


def build_vocabulary(spacy_en, spacy_de):
    def tokenize_en(text):
        return tokenizing(text, spacy_en)

    def tokenize_de(text):
        return tokenizing(text, spacy_de)

    logger.info("Loading datasets ...")
    path_pre = "/data/llm/pretrain_data/wmt14.en-de/processed_data/"
    train_en = path_pre + "train.en.bpe"
    dev_en = path_pre + "dev.en.bpe"
    test_en = path_pre + "test.en.bpe"
    en_files = TextReader([train_en, dev_en, test_en])

    train_de = path_pre + "train.de.bpe"
    dev_de = path_pre + "dev.de.bpe"
    test_de = path_pre + "test.de.bpe"
    de_files = TextReader([train_de, dev_de, test_de])

    train = en_files.dp(train_en).zip(de_files.dp(train_de)).shuffle()
    val = en_files.dp(dev_en).zip(de_files.dp(dev_de))
    test = en_files.dp(test_en).zip(de_files.dp(test_de))

    logger.info("Building English Vocabulary ...")
    vocab_src = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_en, index=0),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    logger.info("Building German Vocabulary ...")
    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_de, index=1),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    return vocab_src, vocab_tgt


def load_vocab(spacy_en, spacy_de):
    path_pre = "/data/llm/pretrain_data/wmt14.en-de/processed_data/"
    file_path = path_pre + "vocab.pt"
    if not exists(file_path):
        vocab_src, vocab_tgt = build_vocabulary(spacy_en, spacy_de)
        torch.save((vocab_src, vocab_tgt), file_path)
    else:
        vocab_src, vocab_tgt = torch.load(file_path)
    logger.info("Finished.\nVocabulary sizes:")
    logger.info(len(vocab_src))
    logger.info(len(vocab_tgt))
    return vocab_src, vocab_tgt


def create_dataloaders(
    device,
    vocab_src,
    vocab_tgt,
    spacy_en,
    spacy_de,
    batch_size=12000,
    max_padding=128,
    is_distributed=True,
):
    def tokenize_en(text):
        return tokenizing(text, spacy_en)

    def tokenize_de(text):
        return tokenizing(text, spacy_de)

    def collate_fn(batch):
        return collate_batch(
            batch,
            tokenize_en,
            tokenize_de,
            vocab_src,
            vocab_tgt,
            device,
            max_padding=max_padding,
            pad_id=vocab_src.get_stoi()["<blank>"],
        )

    logger.info("Loading datasets ...")
    path_pre = "/data/llm/pretrain_data/wmt14.en-de/processed_data/"
    train_en = path_pre + "train.en.bpe"
    dev_en = path_pre + "dev.en.bpe"
    test_en = path_pre + "test.en.bpe"
    en_files = TextReader([train_en, dev_en, test_en])

    train_de = path_pre + "train.de.bpe"
    dev_de = path_pre + "dev.de.bpe"
    test_de = path_pre + "test.de.bpe"
    de_files = TextReader([train_de, dev_de, test_de])

    train_iter = en_files.dp(train_en).zip(de_files.dp(train_de)).shuffle()
    valid_iter = en_files.dp(dev_en).zip(de_files.dp(dev_de))

    train_iter_map = to_map_style_dataset(
        train_iter
    )  # DistributedSampler needs a dataset len()
    train_sampler = (
        DistributedSampler(train_iter_map) if is_distributed else None
    )
    valid_iter_map = to_map_style_dataset(valid_iter)
    valid_sampler = (
        DistributedSampler(valid_iter_map) if is_distributed else None
    )

    train_dataloader = DataLoader(
        train_iter_map,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_iter_map,
        batch_size=batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler,
        collate_fn=collate_fn,
    )
    return train_dataloader, valid_dataloader

def train_worker(
    gpu,
    ngpus_per_node,
    vocab_src,
    vocab_tgt,
    spacy_en,
    spacy_de,
    config,
    is_distributed=False,
):
    logger.info(f"Train worker process using GPU: {gpu} for training", flush=True)
    torch.cuda.set_device(gpu)

    pad_idx = vocab_tgt["<blank>"]
    d_model = 512
    nlayer = 6
    logger.info(
        "Length of vocab_src: {}\nLength of vocab_tgt: {}\n"
        "d_model: {}\nNumber of Layers: {}".format(
            len(vocab_src), len(vocab_tgt), d_model, nlayer
        )
    )
    model = make_model(len(vocab_src), len(vocab_tgt), N=nlayer)
    model.cuda(gpu)
    module = model
    is_main_process = True
    if is_distributed:
        dist.init_process_group(
            "nccl", init_method="env://", rank=gpu, world_size=ngpus_per_node
        )
        model = DDP(model, device_ids=[gpu])
        module = model.module
        is_main_process = (gpu == 0)

    criterion = LabelSmoothing(
        size=len(vocab_tgt), padding_idx=pad_idx, smoothing=0.1
    )
    criterion.cuda(gpu)

    train_dataloader, valid_dataloader = create_dataloaders(
        gpu,
        vocab_src,
        vocab_tgt,
        spacy_en,
        spacy_de,
        batch_size=config["_batch_size"] // ngpus_per_node,
        max_padding=config["max_padding"],
        is_distributed=is_distributed,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["base_lr"], betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, d_model, factor=1, warmup=config["warmup"]
        ),
    )
    train_state = TrainState()

    for epoch in range(config["num_epochs"]):
        if is_distributed:
            train_dataloader.sampler.set_epoch(epoch)
            valid_dataloader.sampler.set_epoch(epoch)

        model.train()
        logger.info(f"[GPU{gpu}] Epoch {epoch} Training ====", flush=True)
        _, train_state = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in train_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train+log",
            accum_iter=config["accum_iter"],
            train_state=train_state,
        )

        if is_main_process:
            file_path = "%s%.2d.pt" % (config["file_prefix"], epoch)
            torch.save(module.state_dict(), file_path)
        torch.cuda.empty_cache()

        logger.info(f"[GPU{gpu}] Epoch {epoch} Validation ====", flush=True)
        model.eval()
        sloss = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in valid_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )
        logger.info(sloss)
        torch.cuda.empty_cache()

    if is_main_process:
        file_path = "%sfinal.pt" % config["file_prefix"]
        torch.save(module.state_dict(), file_path)


def train_distributed_model(vocab_src, vocab_tgt, spacy_en, spacy_de, config):
    ngpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    logger.info(f"Number of GPUs detected: {ngpus}")
    logger.info("Spawning training processes ...")
    mp.spawn(
        train_worker,
        args=(ngpus, vocab_src, vocab_tgt, spacy_en, spacy_de, config, True),
        nprocs=ngpus,
    )


def train_model(vocab_src, vocab_tgt, spacy_en, spacy_de, config):
    if config["distributed"]:
        train_distributed_model(
            vocab_src, vocab_tgt, spacy_en, spacy_de, config
        )
    else:
        train_worker(
            0, 1, vocab_src, vocab_tgt, spacy_en, spacy_de, config, False
        )


def load_trained_model(vocab_src, vocab_tgt, spacy_en, spacy_de):
    config = {
        "_batch_size": 1024,
        "distributed": True,
        "num_epochs": 10,
        "accum_iter": 10,
        "base_lr": 1.0,
        "max_padding": 72,
        "warmup": 4000,
        "file_prefix": "/home/ma-user/work/xiongwen/models/transformer/",
    }
    model_path = config.get("file_prefix") + "wmt14_from_en_to_de.pt"
    if not exists(model_path):
        train_model(vocab_src, vocab_tgt, spacy_en, spacy_de, config)

    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.load_state_dict(torch.load(model_path))
    return model


if __name__ == "__main__":
    logger.info("Running the example .. ")
    # for _ in data_gen(V=13, _batch_size=7, nbatches=3):
    #     print("src info:\n", _.src)
    #     print("src mask info:\n", _.src_mask)
    #     print("tgt info:\n", _.tgt)
    #     print("tgt_y info:\n", _.tgt_y)
    #     print("tgt mask info:\n", _.tgt_mask)
    # example_simple_model()

    spacy_en = load_tokenizer("en")
    spacy_de = load_tokenizer("de")
    vocab_src, vocab_tgt = \
        load_vocab(spacy_de, spacy_en)

    model = load_trained_model(
        vocab_src,
        vocab_tgt,
        spacy_en,
        spacy_de
    )
