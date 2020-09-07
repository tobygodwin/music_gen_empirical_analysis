import torch
from torch import nn
import numpy as np
from librosa.core import load


EPSILON = 1e-2

def linear_quantize(samples, q_levels):
    samples = samples.clone()
    samples -= samples.min(dim=-1)[0].expand_as(samples)
    samples /= samples.max(dim=-1)[0].expand_as(samples)
    samples *= q_levels - EPSILON
    samples += EPSILON / 2
    return samples.long()

def linear_dequantize(samples, q_levels):
    return samples.float() / (q_levels / 2) - 1

def q_zero(q_levels):
    return q_levels // 2

def one_hot(classes, nb_classes) : 
    flat = classes.reshape(-1)
    zeros = torch.zeros(flat.shape[0], nb_classes)
    for i,a in enumerate(flat):
        zeros[i,a] = 1
    shape = [classes.shape[i] for i in range(classes.dim())] + [nb_classes]
    one_hot = zeros.reshape(shape).long()
    return one_hot

def size_splits(tensor, split_sizes, dim=0):
    """Splits the tensor according to chunks of split_sizes.
    
    Arguments:
        tensor (Tensor): tensor to split.
        split_sizes (list(int)): sizes of chunks
        dim (int): dimension along which to split the tensor.
    """
    if dim < 0:
        dim += tensor.dim()
    
    dim_size = tensor.size(dim)
    if dim_size != torch.sum(torch.Tensor(split_sizes)):
        raise KeyError("Sum of split sizes exceeds tensor dim")
    splits = torch.cumsum(torch.Tensor([0] + split_sizes), dim=0)[:-1]
    return tuple(tensor.narrow(int(dim), int(start), int(length)) 
        for start, length in zip(splits, split_sizes))

def load_audio_prompt(file, q_levels, model_lookback, prompt_length):
    # file is full path
    label = int(file.split("/")[-2])  # label associated with file_name
    (seq, sr) = load(file, sr=None, mono=True)
    seq = seq[:prompt_length]
    return torch.cat([
        torch.LongTensor(model_lookback) \
            .fill_(q_zero(q_levels)),
            linear_quantize(
            torch.from_numpy(seq), q_levels
        ),
        torch.LongTensor([label])
    ])