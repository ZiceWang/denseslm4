"""Evaluate DenseSLM4 using LightEval library."""

import os

# Set http(s)_proxy port to 7890 for dataset downloads.
os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"
import json
import torch
from safetensors.torch import load_file

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.transformers.transformers_model import TransformersModel, TransformersModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters

from denseslm4.modeling_denseslm4 import DenseSLM4ForCausalLM
from denseslm4.configuration_denseslm4 import DenseSLM4Config


def create_lighteval_model(checkpoint_path: str, batch_size: int = 1):
    """Load DenseSLM4 model and wrap it for LightEval."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    # Load config from checkpoint
    with open(f"{checkpoint_path}/config.json") as f:
        config_dict = json.load(f)
    
    # Filter config to only include valid DenseSLM4Config parameters
    valid_keys = DenseSLM4Config.__init__.__code__.co_varnames
    filtered_config = {k: v for k, v in config_dict.items() if k in valid_keys}
    config = DenseSLM4Config(**filtered_config)
    
    print(f"Config: vocab_size={config.vocab_size}, hidden_size={config.hidden_size}, "
          f"num_hidden_layers={config.num_hidden_layers}")
    
    # Create model
    model = DenseSLM4ForCausalLM(config)
    
    # Load weights from safetensors
    state_dict = load_file(f"{checkpoint_path}/model.safetensors")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: missing keys: {missing}")
    if unexpected:
        print(f"Warning: unexpected keys: {unexpected}")
    
    # Retie weights (needed for GenerationMixin)
    model._retie_weights()
    
    # Move to CUDA and convert to bfloat16
    model = model.cuda().to(torch.bfloat16)
    model.eval()
    print("Model loaded successfully!")
    
    # Get tokenizer path (same as checkpoint for local tokenizer)
    tokenizer_path = checkpoint_path
    
    # Wrap model for LightEval. LightEval 0.13 uses override_chat_template,
    # not the old use_chat_template argument.
    model_config = TransformersModelConfig(
        model_name=checkpoint_path,
        batch_size=batch_size,
        tokenizer=tokenizer_path,
        max_length=512,
    )
    
    lighteval_model = TransformersModel.from_model(
        model=model,
        config=model_config,
    )
    
    return lighteval_model, tokenizer_path


def evaluate_with_lighteval(
    checkpoint_path: str,
    tasks: str = "truthfulqa:mc|0",
    max_samples: int = None,
    batch_size: int = 1,
    output_dir: str = "./lighteval_results",
):
    """Run lighteval evaluation on a DenseSLM4 checkpoint."""
    
    # Load and wrap model
    model, tokenizer_path = create_lighteval_model(checkpoint_path, batch_size)
    
    # Set up evaluation tracker
    evaluation_tracker = EvaluationTracker(
        output_dir=output_dir,
        save_details=True,
    )
    
    # Set up pipeline parameters
    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.NONE,
        max_samples=max_samples,
    )
    
    # Create pipeline
    pipeline = Pipeline(
        model=model,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        tasks=tasks,
    )
    
    # Run evaluation
    print(f"\nRunning evaluation on tasks: {tasks}")
    results = pipeline.evaluate()
    
    # Show and return results
    pipeline.show_results()
    pipeline.save_and_push_results() # Saves to output_dir
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate DenseSLM4 with LightEval")
    parser.add_argument("--checkpoint", type=str,
                       default="/data1/neu_lab2/denseslm4/runs/my_model_new_tokenizer/final_model",
                       help="Path to model checkpoint")
    parser.add_argument("--tasks", type=str, 
                       default="truthfulqa:mc|0,gsm8k|3",
                       help="Tasks to evaluate (comma-separated)")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Max samples per task (default: all)")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for evaluation")
    parser.add_argument("--output_dir", type=str, default="./lighteval_results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    evaluate_with_lighteval(
        checkpoint_path=args.checkpoint,
        tasks=args.tasks,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )
