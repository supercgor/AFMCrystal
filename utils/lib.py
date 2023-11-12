import torch
import numpy as np
from torch import nn
from numba import njit
from copy import deepcopy

def center(xh):
    x, h = xh[..., :3], xh[..., 3:]
    x = x - x.mean(dim=-2, keepdim=True)
    return torch.cat([x, h], dim=-1)
    
def center_randn_like(xh, *args, **kwargs):
    noise = torch.randn_like(xh, *args, **kwargs)
    noise = noise - noise.mean(dim=-2, keepdim=True)
    return noise

def model_structure(model: nn.Module)-> list[str]:
    out = []
    blank = ' '
    out.append('-' * 100)
    out.append('|' + ' ' * 21 + 'weight name' + ' ' * 20 + '|'
        + ' ' * 10 + 'weight shape' + ' ' * 10 + '|'
        + ' ' * 3 + 'number' + ' ' * 3 + '|')
    out.append('-' * 100)
    num_para = 0
    type_size = 1  # 如果是浮点数就是4

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) > 50:
            key = key.split(".")
            key = ".".join(i[:7] for i in key)
        if len(key) <= 50:
            key = key + (50 - len(key)) * blank
        shape = str(tuple(w_variable.shape))[1:-1]
        if len(shape) <= 30:
            shape = shape + (30 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        out.append('| {} | {} | {} |'.format(key, shape, str_num))
    out.append('-' * 100)
    out.append('The total number of parameters: ' + str(num_para))
    out.append('The parameters of Model {}: {:4f}M'.format(
        model._get_name(), num_para * type_size / 1000 / 1000))
    out.append(f'The memory used now: {torch.cuda.memory_allocated() / 1024 / 1024:.2f}MB')
    out.append('-' * 100)
    return out

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def sum_except_batch(x):
    return x.reshape(x.size(0), -1).sum(dim=-1)


def remove_mean(x):
    mean = torch.mean(x, dim=1, keepdim=True)
    x = x - mean
    return x


def remove_mean_with_mask(x, node_mask: torch.Tensor):
    masked_max_abs_value = (x * (1 - node_mask[...,None])).abs().sum()
    assert masked_max_abs_value < 1e-5, f'Error {masked_max_abs_value.item()} too high'
    N = node_mask[...,None].sum(1, keepdims=True)

    mean = torch.sum(x, dim=1, keepdim=True) / N
    print(mean.shape, node_mask.shape)
    x = x - mean * node_mask[...,None]
    return x


def assert_mean_zero(x):
    mean = torch.mean(x, dim=1, keepdim=True)
    assert mean.abs().max().item() < 1e-4


def assert_mean_zero_with_mask(x, node_mask, eps=1e-10):
    assert_correctly_masked(x, node_mask)
    largest_value = x.abs().max().item()
    error = torch.mean(x, dim=1, keepdim=True).abs().max().item()
    rel_error = error / (largest_value + eps)
    assert rel_error < 1e-2, f'Mean is not zero, relative_error {rel_error}'


def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask[...,None])).abs().max().item() < 1e-4, \
        'Variables not masked properly.'


def center_gravity_zero_gaussian_log_likelihood(x):
    assert len(x.size()) == 3
    B, N, D = x.size()
    assert_mean_zero(x)

    # r is invariant to a basis change in the relevant hyperplane.
    r2 = sum_except_batch(x.pow(2))

    # The relevant hyperplane is (N-1) * D dimensional.
    degrees_of_freedom = (N-1) * D

    # Normalizing constant and logpx are computed:
    log_normalizing_constant = -0.5 * degrees_of_freedom * np.log(2*np.pi)
    log_px = -0.5 * r2 + log_normalizing_constant

    return log_px


def sample_center_gravity_zero_gaussian(size, device):
    assert len(size) == 3
    x = torch.randn(size, device=device)

    # This projection only works because Gaussian is rotation invariant around
    # zero and samples are independent!
    x_projected = remove_mean(x)
    return x_projected


def center_gravity_zero_gaussian_log_likelihood_with_mask(x, node_mask):
    assert len(x.size()) == 3
    B, N_embedded, D = x.size()
    assert_mean_zero_with_mask(x, node_mask)

    # r is invariant to a basis change in the relevant hyperplane, the masked
    # out values will have zero contribution.
    r2 = sum_except_batch(x.pow(2))

    # The relevant hyperplane is (N-1) * D dimensional.
    N = node_mask.squeeze(2).sum(1)  # N has shape [B]
    degrees_of_freedom = (N-1) * D

    # Normalizing constant and logpx are computed:
    log_normalizing_constant = -0.5 * degrees_of_freedom * np.log(2*np.pi)
    log_px = -0.5 * r2 + log_normalizing_constant

    return log_px


def sample_center_gravity_zero_gaussian_with_mask(size, device, node_mask):
    assert len(size) == 3
    x = torch.randn(size, device=device)

    x_masked = x * node_mask[...,None]

    # This projection only works because Gaussian is rotation invariant around
    # zero and samples are independent!
    x_projected = remove_mean_with_mask(x_masked, node_mask)
    return x_projected


def standard_gaussian_log_likelihood(x):
    # Normalizing constant and logpx are computed:
    log_px = sum_except_batch(-0.5 * x * x - 0.5 * np.log(2*np.pi))
    return log_px


def sample_gaussian(size, device):
    x = torch.randn(size, device=device)
    return x


def standard_gaussian_log_likelihood_with_mask(x, node_mask):
    # Normalizing constant and logpx are computed:
    log_px_elementwise = -0.5 * x * x - 0.5 * np.log(2*np.pi)
    log_px = sum_except_batch(log_px_elementwise * node_mask)
    return log_px


def sample_gaussian_with_mask(size, device, node_mask):
    x = torch.randn(size, device=device)
    x_masked = x * node_mask[...,None]
    return x_masked


@torch.jit.script
def __decode_th(positions):
    positions = positions.reshape(-1, 9)
    o, u, v = positions[...,0:3], positions[...,3:6], positions[...,6:]
    u, v = u + v - 2 * o, u - v
    u = u / torch.norm(u, dim=-1, keepdim=True)
    v = v / torch.norm(v, dim=-1, keepdim=True)
    v = torch.where(v[..., 1].unsqueeze(-1) >= 0, v, -v)
    v = torch.where(v[..., 0].unsqueeze(-1) >= 0, v, -v)
    return torch.cat([o, u, v], dim=-1)
    x   
#@nb.njit(fastmath=True, cache=True)
def __decode_np(positions:np.ndarray):
    positions = positions.reshape(-1, 9)
    o, u, v = positions[...,0:3], positions[...,3:6], positions[...,6:]
    u, v = u + v - 2 * o, u - v
    u = u / np.expand_dims(((u**2).sum(axis=-1)**0.5), -1)
    v = v / np.expand_dims(((v**2).sum(axis=-1)**0.5), -1)
    v = np.where(v[..., 1][...,None] >= 0, v, -v)
    v = np.where(v[..., 0][...,None] >= 0, v, -v)
    return np.concatenate((o, u, v), axis=-1)

@torch.jit.script
def __encode_th(emb):
    o, u, v = emb[...,0:3], emb[...,3:6], emb[...,6:]
    h1 = (0.612562225 * u + 0.790422368 * v) * 0.9584 + o
    h2 = (0.612562225 * u - 0.790422368 * v) * 0.9584 + o
    return torch.cat([o, h1, h2], dim=-1)

@njit(fastmath=True,parallel=True, cache=True)
def __encode_np(emb):
    o, u, v = emb[...,0:3], emb[...,3:6], emb[...,6:]
    h1 = (0.612562225 * u + 0.790422368 * v) * 0.9584 + o
    h2 = (0.612562225 * u - 0.790422368 * v) * 0.9584 + o
    return np.concatenate((o, h1, h2), axis=-1)
    
def decodeWater(positions):
    if isinstance(positions, torch.Tensor):
        return __decode_th(positions)
    else:
        return __decode_np(positions)

def encodeWater(emb):
    if isinstance(emb, torch.Tensor):
        return __encode_th(emb)
    else:
        return __encode_np(emb)
    
def model_save(module: nn.Module, path: str):
        try:
            state_dict = module.module.state_dict()
        except AttributeError:
            state_dict = module.state_dict()
        torch.save(state_dict, path)

def model_load(module: nn.Module, path: str, strict = False) -> list[str]:
    state_dict = module.state_dict()
    param_names = list(state_dict.keys())
    pretrained_state_dict = torch.load(path, map_location = 'cpu')
    pretrained_param_names = list(pretrained_state_dict.keys())
    mismatch_list = []
    for i, param in enumerate(pretrained_param_names):
        if i == len(param_names):
            break
        if param == param_names[i]:
            state_dict[param] = pretrained_state_dict[param]
        else:
            mismatch_list.append(param)
            continue
    if strict:
        module.load_state_dict(state_dict)
    else:
        try:
            module.load_state_dict(state_dict)
        except RuntimeError:
            pass
    return mismatch_list

class EMA():
    def __init__(self, beta, model):
        super().__init__()
        self.model = deepcopy(model)
        self.beta = beta

    def update_model_average(self, current_model):
        for current_params, ma_params in zip(current_model.parameters(), self.model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new
