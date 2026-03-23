#!/usr/bin/env python3
"""
Fine-tune a local model on distilled training data from Louter.

Supports LoRA fine-tuning on tool-calling and code generation data,
optimized for small models (1B-8B parameters).

Usage:
    # Basic training on exported data
    python train.py --data training_data.jsonl --base-model Qwen/Qwen2.5-3B-Instruct

    # With custom LoRA config
    python train.py --data training_data.jsonl --base-model meta-llama/Llama-3.2-3B-Instruct \
        --lora-r 32 --lora-alpha 64 --epochs 3

    # Resume from checkpoint
    python train.py --data training_data.jsonl --base-model Qwen/Qwen2.5-3B-Instruct \
        --resume-from ./output/checkpoint-500

    # Export merged model for Ollama
    python train.py --merge --base-model Qwen/Qwen2.5-3B-Instruct \
        --adapter-path ./output --output-dir ./merged_model
"""

import argparse
import json
import sys
from pathlib import Path


def load_jsonl(path: str) -> list[dict]:
    """Load JSONL training data."""
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def prepare_dataset(samples: list[dict], tokenizer):
    """Convert samples to a HuggingFace Dataset with chat template applied."""
    from datasets import Dataset

    processed = []
    for sample in samples:
        messages = sample.get("messages", [])
        if not messages:
            continue

        # Apply chat template
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            processed.append({"text": text})
        except Exception as e:
            print(f"Warning: skipping sample due to template error: {e}", file=sys.stderr)
            continue

    if not processed:
        print("Error: no valid samples after processing", file=sys.stderr)
        sys.exit(1)

    print(f"Prepared {len(processed)} training samples", file=sys.stderr)
    return Dataset.from_list(processed)


def train(args):
    """Run LoRA fine-tuning."""
    import torch
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
    )
    from trl import SFTTrainer

    print(f"Loading base model: {args.base_model}", file=sys.stderr)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load training data
    print(f"Loading training data from: {args.data}", file=sys.stderr)
    samples = load_jsonl(args.data)
    print(f"Loaded {len(samples)} samples", file=sys.stderr)

    dataset = prepare_dataset(samples, tokenizer)

    # Determine dtype and device
    if torch.cuda.is_available():
        dtype = torch.bfloat16
        device_map = "auto"
        print("Using CUDA with bfloat16", file=sys.stderr)
    elif torch.backends.mps.is_available():
        dtype = torch.float16
        device_map = "mps"
        print("Using MPS (Apple Silicon) with float16", file=sys.stderr)
    else:
        dtype = torch.float32
        device_map = "cpu"
        print("Using CPU with float32 (this will be slow)", file=sys.stderr)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    # LoRA configuration
    # Target common attention + MLP modules across architectures
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
    output_dir = args.output_dir or "./output"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to="none",
        remove_unused_columns=False,
    )

    # Resume from checkpoint if specified
    resume_from = args.resume_from

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting training...", file=sys.stderr)
    trainer.train(resume_from_checkpoint=resume_from)

    # Save adapter
    print(f"Saving adapter to {output_dir}", file=sys.stderr)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Training complete!", file=sys.stderr)


def merge_and_export(args):
    """Merge LoRA adapter with base model and export."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading base model: {args.base_model}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )

    adapter_path = args.adapter_path or "./output"
    print(f"Loading adapter from: {adapter_path}", file=sys.stderr)
    model = PeftModel.from_pretrained(model, adapter_path)

    print("Merging adapter with base model...", file=sys.stderr)
    model = model.merge_and_unload()

    output_dir = args.output_dir or "./merged_model"
    print(f"Saving merged model to: {output_dir}", file=sys.stderr)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Create Ollama Modelfile
    modelfile_path = Path(output_dir) / "Modelfile"
    modelfile_path.write_text(
        f'FROM {output_dir}\n'
        f'PARAMETER temperature 0.7\n'
        f'PARAMETER top_p 0.9\n'
        f'SYSTEM "You are a helpful AI assistant specialized in tool calling and code generation."\n'
    )
    print(f"Created Ollama Modelfile at: {modelfile_path}", file=sys.stderr)
    print(
        f"\nTo import into Ollama:\n"
        f"  ollama create louter-distilled -f {modelfile_path}\n",
        file=sys.stderr,
    )


def main():
    parser = argparse.ArgumentParser(description="Louter Distillation Training")
    parser.add_argument(
        "--merge", action="store_true",
        help="Merge LoRA adapter with base model (instead of training)"
    )

    # Model
    parser.add_argument(
        "--base-model", default="Qwen/Qwen2.5-3B-Instruct",
        help="Base model name or path (default: Qwen/Qwen2.5-3B-Instruct)"
    )
    parser.add_argument(
        "--data", help="Path to JSONL training data"
    )
    parser.add_argument(
        "--output-dir", help="Output directory for adapter/merged model"
    )

    # LoRA
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank (default: 16)")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha (default: 32)")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout (default: 0.05)")

    # Training
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--gradient-accumulation", type=int, default=4, help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    parser.add_argument("--save-steps", type=int, default=100, help="Save checkpoint every N steps")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing")

    # Merge
    parser.add_argument("--adapter-path", help="Path to LoRA adapter (for --merge)")

    # Resume
    parser.add_argument("--resume-from", help="Resume training from checkpoint")

    args = parser.parse_args()

    if args.merge:
        merge_and_export(args)
    else:
        if not args.data:
            print("Error: --data is required for training", file=sys.stderr)
            sys.exit(1)
        train(args)


if __name__ == "__main__":
    main()
