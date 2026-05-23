"""Write the DenseSLM4 chat template into a tokenizer directory."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from transformers import AutoTokenizer

from denseslm4.sft_moe import ensure_chat_template


def main(
	model_or_tokenizer_path: Annotated[Path, typer.Argument(help="Model/tokenizer directory to update.")],
) -> None:
	tokenizer = AutoTokenizer.from_pretrained(model_or_tokenizer_path)
	ensure_chat_template(tokenizer)
	tokenizer.save_pretrained(model_or_tokenizer_path)
	type_name = type(tokenizer).__name__
	typer.echo(f"Saved DenseSLM4 chat_template to {model_or_tokenizer_path} ({type_name}).")


if __name__ == "__main__":
	typer.run(main)
