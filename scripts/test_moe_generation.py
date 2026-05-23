"""Test generation script for DenseSLM4MoE."""

import torch
from transformers import AutoTokenizer
from safetensors.torch import load_file
import json
import sys
import argparse
from threading import Thread

sys.path.insert(0, "src")
from denseslm4.configuration_denseslm4moe import DenseSLM4MoeConfig
from denseslm4.modeling_denseslm4moe import DenseSLM4MoeForCausalLM
from transformers import TextIteratorStreamer


def main():
    parser = argparse.ArgumentParser(description="Test MoE model generation")
    parser.add_argument("--checkpoint", type=str, default="./runs/denseslm4_moe/final_model",
                        help="Path to checkpoint")
    parser.add_argument("--prompt", type=str, default="Once upon a time",
                        help="Prompt for generation")
    parser.add_argument("--max_new_tokens", type=int, default=100,
                        help="Maximum new tokens to generate")
    parser.add_argument("--stream", action="store_true", default=True,
                        help="Stream generated text as it is produced")
    parser.add_argument("--no-stream", dest="stream", action="store_false",
                        help="Disable streaming and print the final decoded text")
    args = parser.parse_args()

    # Create model
    model = DenseSLM4MoeForCausalLM.from_pretrained(args.checkpoint)
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

    generation_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    if args.stream:
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        print("\nGenerated:")

        def generate_in_background() -> None:
            with torch.no_grad():
                model.generate(**generation_kwargs, streamer=streamer)

        thread = Thread(target=generate_in_background)
        thread.start()
        generated_text = args.prompt
        for new_text in streamer:
            generated_text += new_text
            print(new_text, end="", flush=True)
        thread.join()
        print()
        return

    with torch.no_grad():
        output_ids = model.generate(**generation_kwargs)

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"\nGenerated:\n{generated_text}")


if __name__ == "__main__":
    main()
