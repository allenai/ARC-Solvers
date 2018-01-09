"""
Assorted utilities
"""
import torch
from allennlp.common.checks import ConfigurationError
from allennlp.nn.util import replace_masked_values


def masked_mean(tensor, dim, mask):
    """
    ``Performs a mean on just the non-masked portions of the ``tensor`` in the
    ``dim`` dimension of the tensor.
    """
    if mask is None:
        return torch.mean(tensor, dim)
    if tensor.dim() != mask.dim():
        raise ConfigurationError("tensor.dim() (%d) != mask.dim() (%d)" % (tensor.dim(), mask.dim()))
    masked_tensor = replace_masked_values(tensor, mask, 0.0)
    # total value
    total_tensor = torch.sum(masked_tensor, dim)
    # count
    count_tensor = torch.sum((mask != 0), dim)
    # set zero count to 1 to avoid nans
    zero_count_mask = (count_tensor == 0)
    count_plus_zeros = (count_tensor + zero_count_mask).float()
    # average
    mean_tensor = total_tensor / count_plus_zeros
    return mean_tensor


