def transfer_to_gpu(var_iterable):
    """
    Transfer torch variables/tensor to GPU
    :param var_iterable:
    :return: list of GPU variabels/tensor
    """
    return list(v.cuda() for v in var_iterable)