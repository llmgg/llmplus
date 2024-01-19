# Count some parameters used in LLM


def count_model_size(model_config=dict()):
    """
    Count the model size
    """
    l = model_config.get('num_layers', 32)
    h = model_config.get('num_hidden', 4096)
    v = model_config.get('num_vocab', 32000)

    size = 12.0 * l * h ** 2 + (2.0 * l + 1) * h + 2 * h * v

    return size / 1.0e9


def count_compute_time(model_config=dict(), seq_len=1000, ntokens=1.0, ngpu=8, gpu_type='V100', ratio=0.5):
    T = ntokens*1.0e9
    s = seq_len
    l = model_config.get('num_layers', 32)
    h = model_config.get('num_hidden', 4096)
    v = model_config.get('num_vocab', 32000)

    total_time = 72.0*T*l*h**2*(1.0 + s/(6.0*h) + v/(12.0*l*h))

    FT_default = 1.0e12
    if gpu_type == 'V100':
        FT_default = 112e12
    elif gpu_type == 'A100':
        FT_default = 312e12

    return total_time / (ngpu*FT_default*ratio*24*3600)


if __name__ == "__main__":
    model_config = dict(
        num_layers=32,
        num_hidden=4096,
        num_vocab=32000,
    )
    print(count_model_size(model_config))

    print(count_compute_time(model_config, seq_len=1000, ntokens=5.0, ratio=0.5))

