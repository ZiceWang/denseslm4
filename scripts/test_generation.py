"""Test script to load a checkpoint and generate text using GenerationMixin."""

import torch
from transformers import AutoTokenizer  # noqa: F401 - needed for tokenizer loading
from safetensors.torch import load_file
from denseslm4.modeling_denseslm4 import DenseSLM4ForCausalLM
from denseslm4.configuration_denseslm4 import DenseSLM4Config


def generate_from_checkpoint(checkpoint_path: str, prompt: str, max_new_tokens: int = 50):
    """Load a checkpoint and generate text from a prompt."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    # Load config directly from file
    import json
    with open(f"{checkpoint_path}/config.json") as f:
        config_dict = json.load(f)
    config = DenseSLM4Config(**{k: v for k, v in config_dict.items() 
                          if k in DenseSLM4Config.__init__.__code__.co_varnames})
    print(f"Config: vocab_size={config.vocab_size}, hidden_size={config.hidden_size}, "
          f"num_hidden_layers={config.num_hidden_layers}, num_attention_heads={config.num_attention_heads}")
    
    # Create model
    model = DenseSLM4ForCausalLM(config)
    
    # Load weights manually (from_pretrained doesn't work for unregistered models)
    state_dict = load_file(f"{checkpoint_path}/model.safetensors")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: missing keys: {missing}")
    if unexpected:
        print(f"Warning: unexpected keys: {unexpected}")
    
    # Retie weights after loading
    model._retie_weights()
    
    model = model.cuda().to(torch.bfloat16)
    model.eval()
    print(f"Model loaded successfully!")
    
    # Load tokenizer from local files
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize prompt
    print(f"\nPrompt: {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    print(f"Input tokens: {input_ids.shape[1]}")
    
    # Generate
    print("Generating...")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False,  # Disable cache due to Mamba2 state cache incompatibility
        )
    
    # Decode
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"\nGenerated:\n{generated_text}")
    
    return generated_text


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test generation from checkpoint")
    parser.add_argument("--checkpoint", type=str, 
                       default="runs/tinystories/final_model",
                       help="Path to checkpoint")
    parser.add_argument("--prompt", type=str, default="Once upon a time",
                       help="Prompt for generation")
    parser.add_argument("--max_new_tokens", type=int, default=50,
                       help="Maximum new tokens to generate")
    args = parser.parse_args()
    
    generate_from_checkpoint(args.checkpoint, args.prompt, args.max_new_tokens)
