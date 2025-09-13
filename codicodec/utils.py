import os
import torch
import numpy as np

from .hparams import *
from .audio import *

try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None


# Scaling coefficients used to condition the denoiser while satisfying boundary conditions
# Returns broadcasting-friendly tensors matching input rank (B, 1, 1, 1) or (B, 1, 1, T)
def get_c(sigma):
    """Compute c_skip, c_out, c_in given noise level sigma.

    Args:
        sigma: float, 1D or 2D tensor of noise scales.
    Returns:
        Tuple of tensors (c_skip, c_out, c_in) broadcastable to spectrograms.
    """
    sigma_correct = sigma_min
    c_skip = (sigma_data**2.)/(((sigma - sigma_correct)**2.) + (sigma_data**2.))
    c_out = (sigma_data * (sigma - sigma_correct)) / (((sigma_data**2.) + (sigma**2.))**0.5)
    c_in = 1. / (((sigma**2.) + (sigma_data**2.))**0.5)
    if len(sigma.shape) == 1:
        return c_skip.reshape(-1, 1, 1, 1), c_out.reshape(-1, 1, 1, 1), c_in.reshape(-1, 1, 1, 1)
    elif len(sigma.shape) == 2:
        return (
            c_skip.reshape(sigma.shape[0], 1, 1, -1),
            c_out.reshape(sigma.shape[0], 1, 1, -1),
            c_in.reshape(sigma.shape[0], 1, 1, -1),
        )
    else:
        return c_skip, c_out, c_in


# Continuous noise schedule used by the decoder
def get_sigma_continuous(i):
    """Map a continuous index i in [0, 1] to a noise sigma.

    Follows parameterization from https://openreview.net/pdf?id=FmqFfMTNnv
    """
    return (sigma_min**(1./rho) + i * (sigma_max**(1./rho) - sigma_min**(1./rho)))**rho


def get_sigma_step_continuous(sigma_i, step):
    """Lower sigma_i by an absolute step in the continuous schedule.

    Args:
        sigma_i: current sigma.
        step: absolute step size in schedule space.
    """
    return ((sigma_i**(1./rho) - step * (sigma_max**(1./rho) - sigma_min**(1./rho)))**rho).clamp(min=sigma_min)


def add_noise(x, noise, sigma):
    """Add Gaussian noise with scale sigma to x.

    Supports scalar float/int, 1D per-sample, or 2D per-time sigma tensors.
    """
    if isinstance(sigma, int):
        sigma = float(sigma)
    if isinstance(sigma, float):
        sigma = torch.tensor(sigma, device=x.device)
    if len(sigma.shape) == 1:
        sigma = sigma.reshape(-1, 1, 1, 1)
    elif len(sigma.shape) == 2:
        sigma = sigma.reshape(sigma.shape[0], 1, 1, -1)
    return x + sigma * noise


def get_step_continuous(inds, step):
    """Subtract a fixed step from continuous indices and clamp to [0, 1]."""
    steps = torch.ones_like(inds) * step
    return (inds - steps).clamp(min=0.)


def reverse_step(x, noise, sigma):
    """One probability-flow ODE step from sigma to the next lower sigma.

    x_{t-1} = x_t + (sigma^2 - sigma_min^2)^{1/2} * noise
    """
    if isinstance(sigma, int):
        sigma = float(sigma)
    if isinstance(sigma, float):
        sigma = torch.tensor(sigma, device=x.device)
    if len(sigma.shape) == 1:
        sigma = sigma.reshape(-1, 1, 1, 1)
    elif len(sigma.shape) == 2:
        sigma = sigma.reshape(sigma.shape[0], 1, 1, -1)
    return x + ((sigma**2 - sigma_min**2)**0.5) * noise


@torch.no_grad()
def distribute(model, x, max_batch_size, device, *args, **kwargs):
    """Apply model to x by splitting to multiple batches with max_batch_size.

    Moves inputs to the specified device, optionally autocasts to fp16
    if mixed_precision is enabled, and stitches outputs back together.
    """
    data_device = x.device

    def split_tensor(t, batch_size):
        return torch.split(t, batch_size, dim=0)

    def split_arg(arg, batch_size):
        if isinstance(arg, torch.Tensor):
            return split_tensor(arg, batch_size)
        elif isinstance(arg, list):
            if any(isinstance(item, torch.Tensor) for item in arg):
                splits = [split_tensor(item, batch_size) if isinstance(item, torch.Tensor) else [item] * num_batches
                          for item in arg]
                return [[split[i] for split in splits] for i in range(num_batches)]
            return [arg] * num_batches
        return [arg] * num_batches

    def to_device(arg):
        if isinstance(arg, torch.Tensor):
            return arg.to(device)
        elif isinstance(arg, list):
            return [to_device(a) for a in arg]
        return arg

    if max_batch_size is None or x.shape[0] <= max_batch_size:
        x = x.to(device)
        args = tuple(to_device(arg) for arg in args)
        kwargs = {k: to_device(v) for k, v in kwargs.items()}

        with torch.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', dtype=torch.float16, enabled=mixed_precision):
            outputs = model(x, *args, **kwargs)
        if isinstance(outputs, list):
            outputs = [out.to(data_device) for out in outputs]
        else:
            outputs = outputs.to(data_device)
    else:
        num_batches = (x.shape[0] + max_batch_size - 1) // max_batch_size
        x_splits = split_tensor(x, max_batch_size)

        arg_splits = [split_arg(arg, max_batch_size) for arg in args]
        kwarg_splits = {k: split_arg(v, max_batch_size) for k, v in kwargs.items()}

        outputs = []
        for i in range(num_batches):
            batch_x = x_splits[i].to(device)
            batch_args = tuple(to_device(arg_split[i]) for arg_split in arg_splits)
            batch_kwargs = {k: to_device(kwarg_splits[k][i]) for k in kwargs}

            with torch.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', dtype=torch.float16, enabled=mixed_precision):
                batch_output = model(batch_x, *batch_args, *batch_kwargs)
            if isinstance(batch_output, list):
                batch_output = [out.to(data_device) for out in batch_output]
            else:
                batch_output = batch_output.to(data_device)
            outputs.append(batch_output)

        if isinstance(outputs[0], list):
            outputs = [torch.cat([out[j] for out in outputs], dim=0) for j in range(len(outputs[0]))]
        else:
            outputs = torch.cat(outputs, dim=0)

    return outputs


def preprocess_parallel_input(x, iteration, length, dim=-1):
    """Prepare inputs for parallel decoding by interleaving context slots."""
    x = torch.split(x, length, dim=dim)
    if iteration % 2 != 0:
        x = list(x)
        x.insert(1, torch.zeros_like(x[0]))
        x.append(torch.zeros_like(x[0]))
    x = torch.cat(x, dim=dim)
    x = torch.split(x, length * 2, dim=dim)
    return torch.cat(x, dim=0), len(x)


def preprocess_parallel_features(x, iteration, num_samples, dim=-1):
    """Prepare feature lists for parallel decoding schedule."""
    x = [torch.chunk(el, num_samples, dim=0) for el in x]
    if iteration % 2 != 0:
        num_samples += 2
        for i, el in enumerate(x):
            el = list(el)
            el.insert(1, torch.zeros_like(el[0]))
            el.append(torch.zeros_like(el[0]))
            x[i] = el
    x = [torch.cat(el, dim=dim) for el in x]
    x = [torch.chunk(el, num_samples // 2, dim=dim) for el in x]
    x = [torch.cat(el, dim=0) for el in x]
    return x


def postprocess_parallel_input(x, iteration, num_samples, length, dim=-1):
    """Undo interleaving to restore original layout after a parallel step."""
    x = torch.chunk(x, num_samples, dim=0)
    x = torch.cat(x, dim=dim)
    x = torch.split(x, length, dim=dim)
    if iteration % 2 != 0:
        x = list(x)
        x.pop(1)
        x.pop()
    return torch.cat(x, dim=dim)


def is_integer(x):
    """Return True if x is an integer-valued np.ndarray or torch.Tensor."""
    if isinstance(x, np.ndarray):
        return np.issubdtype(x.dtype, np.integer)
    elif isinstance(x, torch.Tensor):
        return x.dtype in [torch.int32, torch.int64]
    return False


def is_path(variable):
    """Return True if variable is a filesystem path that exists."""
    return isinstance(variable, str) and os.path.exists(variable)


def download_model():
    """Download the codicodec checkpoint from the Hugging Face Hub if missing.

    Expects a file named 'codicodec.pt' in the local 'models' directory inside
    the installed package. The repo_id is assumed to be 'SonyCSLParis/codicodec'.
    """
    filepath = os.path.abspath(__file__)
    lib_root = os.path.dirname(filepath)
    local_dir = os.path.join(lib_root, "models")
    local_path = os.path.join(local_dir, "codicodec.pt")

    if os.path.exists(local_path):
        return

    os.makedirs(local_dir, exist_ok=True)
    if hf_hub_download is None:
        raise RuntimeError("huggingface_hub is required to download the model.")
    print("Downloading model...")
    downloaded_path = hf_hub_download(
        repo_id="SonyCSLParis/codicodec",
        filename="codicodec.pt",
        cache_dir=local_dir,
        local_dir=local_dir,
    )
    # If the file was saved under a nested path, copy/move it into expected location
    if downloaded_path != local_path and os.path.exists(downloaded_path):
        try:
            # Avoid overwriting if same inode
            if not os.path.exists(local_path):
                import shutil
                shutil.copy2(downloaded_path, local_path)
        except Exception:
            pass
    print("Model was downloaded successfully!")