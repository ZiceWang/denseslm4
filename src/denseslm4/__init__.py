"""DenseSLM4 package exports."""

from .configuration_denseslm4 import DenseSLM4Config
from .modeling_denseslm4 import DenseSLM4ForCausalLM, DenseSLM4Model, DenseSLM4PreTrainedModel

__all__ = [
    "DenseSLM4Config",
    "DenseSLM4ForCausalLM",
    "DenseSLM4Model",
    "DenseSLM4PreTrainedModel",
]


def main() -> None:
    print("Hello from denseslm4!")
