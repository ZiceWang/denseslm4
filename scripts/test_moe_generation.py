"""Test generation script for DenseSLM4MoE."""

import torch
from transformers import AutoTokenizer
from safetensors.torch import load_file
import json
import sys
import argparse

sys.path.insert(0, "src")
from denseslm4.configuration_denseslm4moe import DenseSLM4MoeConfig
from denseslm4.modeling_denseslm4moe import DenseSLM4MoeForCausalLM


def main():
    parser = argparse.ArgumentParser(description="Test MoE model generation")
    parser.add_argument("--checkpoint", type=str, default="runs/denseslm4_moe/final_model",
                        help="Path to checkpoint")
    parser.add_argument("--prompt", type=str, default="Once upon a time",
                        help="Prompt for generation")
    parser.add_argument("--max_new_tokens", type=int, default=100,
                        help="Maximum new tokens to generate")
    args = parser.parse_args()

    # Load config
    with open(f"{args.checkpoint}/config.json") as f:
        config_dict = json.load(f)
    config = DenseSLM4MoeConfig(**{k: v for k, v in config_dict.items()
                            if k in DenseSLM4MoeConfig.__init__.__code__.co_varnames})
    print(f"Config: vocab_size={config.vocab_size}, hidden_size={config.hidden_size}, "
          f"num_hidden_layers={config.num_hidden_layers}")

    # Create model
    model = DenseSLM4MoeForCausalLM(config)
    state_dict = load_file(f"{args.checkpoint}/model.safetensors")
    model.load_state_dict(state_dict, strict=False)
    model._retie_weights()
    model = model.cuda().to(torch.bfloat16)
    model.eval()
    print("Model loaded!")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Generate
    inputs = tokenizer(args.prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    print(f"\nPrompt: {args.prompt}")
    print(f"Input tokens: {input_ids.shape[1]}")

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"\nGenerated:\n{generated_text}")


if __name__ == "__main__":
    main()
