from torch.utils.data import Dataset
from torchdata.datapipes.iter import FileOpener, IterableWrapper, StreamReader


class TextReader(Dataset):
    """
    Read Text from dist
    """
    def __init__(self, data_paths: list, mode="r", split='\n', batch_size=128):
        self.data_paths_dp = IterableWrapper(data_paths)
        self.mode = mode
        self.split = split
        self.batch_size = batch_size
        self.file_dp = FileOpener(self.data_paths_dp, mode=self.mode)
        self.stream_dp = StreamReader(self.file_dp)
        self.data_buffer = dict()
        self.create_dp()

    def create_dp(self):
        for stream in self.stream_dp:
            file_name = stream[0]
            data = stream[1].split(self.split)
            self.data_buffer.update({file_name: data})
            print("File Path: ", file_name)
            print("Number of Lines: ", len(data))

    def __getitem__(self, path):
        return self.data_buffer.get(path)

    def dp(self, path):
        return IterableWrapper(self.__getitem__(path))


if __name__ == "__main__":
    paths = \
        [
            "/data/llm/pretrain_data/wmt14.en-de/processed_data/train.en.bpe",
            "/data/llm/pretrain_data/wmt14.en-de/processed_data/dev.en.bpe",
            "/data/llm/pretrain_data/wmt14.en-de/processed_data/test.en.bpe",
        ]

    tr = TextReader(paths)
    print(len(tr.dp(paths[0]) + tr.dp(paths[1]) + tr.dp(paths[2])))
    for path in paths:
        print("path: ", path)
        print(len(tr.dp(path)))
        iline = 0
        for _ in tr.dp(path):
            print(_)
            iline += 1
            if iline > 10:
                break
