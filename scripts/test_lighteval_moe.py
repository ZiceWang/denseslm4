"""Evaluate DenseSLM4MoE using LightEval library."""

import argparse
import json

import torch
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.transformers.transformers_model import TransformersModel, TransformersModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from safetensors.torch import load_file

from denseslm4.configuration_denseslm4moe import DenseSLM4MoeConfig
from denseslm4.modeling_denseslm4moe import DenseSLM4MoeForCausalLM


def create_lighteval_model(checkpoint_path: str, batch_size: int = 1):
    """Load DenseSLM4MoE model and wrap it for LightEval."""
    print(f"Loading checkpoint from {checkpoint_path}...")

    with open(f"{checkpoint_path}/config.json") as f:
        config_dict = json.load(f)

    valid_keys = DenseSLM4MoeConfig.__init__.__code__.co_varnames
    filtered_config = {k: v for k, v in config_dict.items() if k in valid_keys}
    config = DenseSLM4MoeConfig(**filtered_config)

    print(
        f"Config: vocab_size={config.vocab_size}, hidden_size={config.hidden_size}, "
        f"num_hidden_layers={config.num_hidden_layers}"
    )

    model = DenseSLM4MoeForCausalLM(config)
    state_dict = load_file(f"{checkpoint_path}/model.safetensors")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: missing keys: {missing}")
    if unexpected:
        print(f"Warning: unexpected keys: {unexpected}")
    model._retie_weights()

    model = model.cuda().to(torch.bfloat16)
    model.eval()
    print("Model loaded successfully!")

    model_config = TransformersModelConfig(
        model_name=checkpoint_path,
        batch_size=batch_size,
        tokenizer=checkpoint_path,
        max_length=512,
    )
    lighteval_model = TransformersModel.from_model(model=model, config=model_config)
    return lighteval_model


def evaluate_with_lighteval(
    checkpoint_path: str,
    tasks: str,
    max_samples: int | None,
    batch_size: int,
    output_dir: str,
):
    model = create_lighteval_model(checkpoint_path, batch_size)

    evaluation_tracker = EvaluationTracker(
        output_dir=output_dir,
        save_details=True,
    )
    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.NONE,
        max_samples=max_samples,
        load_tasks_multilingual=True,
    )
    pipeline = Pipeline(
        model=model,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        tasks=tasks,
    )

    print(f"\nRunning evaluation on tasks: {tasks}")
    results = pipeline.evaluate()
    pipeline.show_results()
    pipeline.save_and_push_results()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DenseSLM4MoE with LightEval")
    parser.add_argument("--checkpoint", type=str, default="./runs/denseslm4_moe/final_model")
    parser.add_argument("--tasks", type=str, default="ceval_zho_mcf|0")
    parser.add_argument("--max_samples", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="./lighteval_results_moe")
    args = parser.parse_args()

    evaluate_with_lighteval(
        checkpoint_path=args.checkpoint,
        tasks=args.tasks,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
