import torch
import math
def pos_embedding(pos, dim, max_period=10000):
    
    """
    Create sinusoidal timestep embeddings.

    :param timesteps(int64): a 1-D Tensor of N indices, one per batch element.
                    These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    data_dim = pos.shape[-1]
    data_dim = dim // (data_dim * 2)
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=data_dim, dtype=torch.float, device=pos.device) / data_dim) # (1, 1, data_dim)
    args = pos[..., None] * freqs.view((1,)*len(pos.shape[:-1]) + (-1,))
    embedding = torch.cat([args.cos(), args.sin()], dim=-1).flatten(-2)
    if dim % data_dim:
        embedding = torch.cat([embedding, torch.zeros(*pos.shape[:-1], dim % data_dim, device = pos.device)], dim=-1)
    return embedding