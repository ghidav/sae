import os
from typing import Any, Type, TypeVar, cast, Iterable, Dict

import torch
from functools import partial
from accelerate.utils import send_to_device
from torch import Tensor, nn
from torch.utils.data import DataLoader
from transformers import PreTrainedModel
from .config import TrainConfig

T = TypeVar("T")

# Iterator
class CycleIterator:
    """An iterator that cycles through an iterable indefinitely.

    Example:
        >>> iterator = CycleIterator([1, 2, 3])
        >>> [next(iterator) for _ in range(5)]
        [1, 2, 3, 1, 2]

    Note:
        Unlike ``itertools.cycle``, this iterator does not cache the values of the iterable.
    """

    def __init__(self, iterable: Iterable) -> None:
        self.iterable = iterable
        self.epoch = 0
        self._iterator = None

    def __next__(self) -> Any:
        if self._iterator is None:
            self._iterator = iter(self.iterable)
        try:
            return next(self._iterator)
        except StopIteration:
            print(f"CycleIterator: Epoch {self.epoch} completed.")
            self._iterator = iter(self.iterable)
            self.epoch += 1
            return next(self._iterator)

    def __iter__(self) -> "CycleIterator":
        return self


def assert_type(typ: Type[T], obj: Any) -> T:
    """Assert that an object is of a given type at runtime and return it."""
    if not isinstance(obj, typ):
        raise TypeError(f"Expected {typ.__name__}, got {type(obj).__name__}")

    return cast(typ, obj)


@torch.no_grad()
def geometric_median(points: Tensor, max_iter: int = 100, tol: float = 1e-5):
    """Compute the geometric median `points`. Used for initializing decoder bias."""
    # Initialize our guess as the mean of the points
    guess = points.mean(dim=0)
    prev = torch.zeros_like(guess)

    # Weights for iteratively reweighted least squares
    weights = torch.ones(len(points), device=points.device)

    for _ in range(max_iter):
        prev = guess

        # Compute the weights
        weights = 1 / torch.norm(points - guess, dim=1)

        # Normalize the weights
        weights /= weights.sum()

        # Compute the new geometric median
        guess = (weights.unsqueeze(1) * points).sum(dim=0)

        # Early stopping condition
        if torch.norm(guess - prev) < tol:
            break

    return guess


def get_layer_list(model: PreTrainedModel) -> tuple[str, nn.ModuleList]:
    """Get the list of layers to train SAEs on."""
    N = assert_type(int, model.config.num_hidden_layers)
    candidates = [
        (name, mod)
        for (name, mod) in model.named_modules()
        if isinstance(mod, nn.ModuleList) and len(mod) == N
    ]
    assert len(candidates) == 1, "Could not find the list of layers."

    return candidates[0]


@torch.inference_mode()
def resolve_widths(
    cfg: TrainConfig,
    model: PreTrainedModel,
    module_names: list[str],
    dim: int = -1,
    dl: DataLoader | None = None,
) -> dict[str, int]:
    """Find number of output dimensions for the specified modules."""
    module_to_name = {model.get_submodule(name): name for name in module_names}
    hidden_dict: Dict[str, Tensor] = {}

    def standard_hook(
        module: nn.Module,
        _,
        outputs,
        module_to_name: Dict[nn.Module, str],
        hidden_dict: Dict[str, Tensor],
    ):
        # Maybe unpack tuple outputs
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        name = module_to_name[module]
        hidden_dict[name] = outputs.flatten(0, 1)

    hook = partial(
        standard_hook,
        module_to_name=module_to_name,
        hidden_dict=hidden_dict,
    )

    handles = [mod.register_forward_hook(hook) for mod in module_to_name]
    if dl is None:
        dummy = model.dummy_inputs
    else:
        dummy = next(iter(dl))
    dummy = send_to_device(dummy, model.device)
    try:
        model(**dummy)
    finally:
        for handle in handles:
            handle.remove()

    shapes = {name: hidden.shape[dim] for name, hidden in hidden_dict.items()}
    return shapes


# Fallback implementation of SAE decoder
def eager_decode(top_indices: Tensor, top_acts: Tensor, W_dec: Tensor):
    buf = top_acts.new_zeros(top_acts.shape[:-1] + (W_dec.shape[-1],))
    acts = buf.scatter_(dim=-1, index=top_indices, src=top_acts)
    return acts @ W_dec.mT


# Triton implementation of SAE decoder
def triton_decode(top_indices: Tensor, top_acts: Tensor, W_dec: Tensor):
    return TritonDecoder.apply(top_indices, top_acts, W_dec)


try:
    from .kernels import TritonDecoder
except ImportError:
    decoder_impl = eager_decode
    print("Triton not installed, using eager implementation of SAE decoder.")
else:
    if os.environ.get("SAE_DISABLE_TRITON") == "1":
        print("Triton disabled, using eager implementation of SAE decoder.")
        decoder_impl = eager_decode
    else:
        decoder_impl = triton_decode