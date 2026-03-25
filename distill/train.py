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

    # Convert to GGUF if llama-cpp-conversions are available
    gguf_path = convert_to_gguf(output_dir, args.gguf_type)

    if gguf_path:
        # Create Ollama Modelfile pointing to GGUF
        modelfile_path = Path(output_dir) / "Modelfile"
        modelfile_path.write_text(
            f'FROM {gguf_path}\n'
            f'PARAMETER temperature 0.7\n'
            f'PARAMETER top_p 0.9\n'
            f'TEMPLATE """{{{{- range .Messages }}}}{{{{- if eq .Role "system" }}}}<|im_start|>system\n{{{{ .Content }}}}<|im_end|>\n{{{{- else if eq .Role "user" }}}}<|im_start|>user\n{{{{ .Content }}}}<|im_end|>\n{{{{- else if eq .Role "assistant" }}}}<|im_start|>assistant\n{{{{ .Content }}}}<|im_end|>\n{{{{- end }}}}{{{{- end }}}}<|im_start|>assistant\n"""\n'
            f'SYSTEM "You are a helpful AI assistant specialized in tool calling and code generation."\n'
        )
        print(f"Created Ollama Modelfile at: {modelfile_path}", file=sys.stderr)
        print(
            f"\nTo import into Ollama:\n"
            f"  ollama create louter-distilled -f {modelfile_path}\n",
            file=sys.stderr,
        )
    else:
        print(
            f"\nMerged model saved to: {output_dir}\n"
            f"\nTo deploy to Ollama, convert to GGUF first:\n"
            f"  pip install llama-cpp-python\n"
            f"  # Or use llama.cpp's convert_hf_to_gguf.py:\n"
            f"  git clone https://github.com/ggerganov/llama.cpp\n"
            f"  python llama.cpp/convert_hf_to_gguf.py {output_dir} --outfile {output_dir}/model.gguf --outtype q4_k_m\n"
            f"\n"
            f"Then create a Modelfile:\n"
            f"  echo 'FROM {output_dir}/model.gguf' > {output_dir}/Modelfile\n"
            f"  ollama create louter-distilled -f {output_dir}/Modelfile\n",
            file=sys.stderr,
        )


def convert_to_gguf(model_dir: str, quantization: str = "q4_k_m") -> str | None:
    """Convert a HuggingFace model to GGUF format using llama.cpp.

    Ollama only supports direct safetensors import for Llama, Mistral, Gemma,
    and Phi3 architectures. Qwen and other architectures need GGUF conversion.

    Returns the path to the GGUF file, or None if conversion tools are not available.
    """
    import shutil
    import subprocess

    gguf_path = str(Path(model_dir) / f"model-{quantization}.gguf")

    # Try to find convert_hf_to_gguf.py from llama.cpp
    # Check common locations: sibling dir, PATH, or llama-cpp-python install
    convert_script = None
    search_paths = [
        Path(model_dir).parent / "llama.cpp" / "convert_hf_to_gguf.py",
        Path.home() / "llama.cpp" / "convert_hf_to_gguf.py",
    ]

    for p in search_paths:
        if p.exists():
            convert_script = str(p)
            break

    # Also check if it's on PATH
    if not convert_script and shutil.which("convert_hf_to_gguf.py"):
        convert_script = "convert_hf_to_gguf.py"

    if not convert_script:
        print(
            "Note: llama.cpp not found — skipping GGUF conversion.\n"
            "To enable automatic conversion, clone llama.cpp next to the distill directory:\n"
            "  git clone https://github.com/ggerganov/llama.cpp ../llama.cpp\n"
            "  pip install -r ../llama.cpp/requirements.txt",
            file=sys.stderr,
        )
        return None

    print(f"Converting to GGUF ({quantization})...", file=sys.stderr)
    try:
        subprocess.run(
            [
                sys.executable, convert_script,
                model_dir,
                "--outfile", gguf_path,
                "--outtype", quantization,
            ],
            check=True,
        )
        print(f"GGUF model saved to: {gguf_path}", file=sys.stderr)
        return gguf_path
    except subprocess.CalledProcessError as e:
        print(f"Warning: GGUF conversion failed: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(description="Louter Distillation Training")
    parser.add_argument(
        "--merge", action="store_true",
        help="Merge LoRA adapter with base model (instead of training)"
    )

    # Model
    parser.add_argument(
        "--base-model", default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Base model name or path (default: Qwen/Qwen2.5-1.5B-Instruct)"
    )
    parser.add_argument(
        "--data", help="Path to JSONL training data"
    )
    parser.add_argument(
        "--output-dir", help="Output directory for adapter/merged model"
    )

    # LoRA — smaller rank for 1.5B model (less parameters to adapt)
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank (default: 8)")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha (default: 16)")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout (default: 0.05)")

    # Training — larger batch for small model, more epochs for less data
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--gradient-accumulation", type=int, default=2, help="Gradient accumulation steps (default: 2)")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate (default: 5e-4)")
    parser.add_argument("--save-steps", type=int, default=50, help="Save checkpoint every N steps")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing")

    # Merge
    parser.add_argument("--adapter-path", help="Path to LoRA adapter (for --merge)")
    parser.add_argument(
        "--gguf-type", default="q4_k_m",
        help="GGUF quantization type (default: q4_k_m). Options: f16, q8_0, q4_k_m, q4_0, etc."
    )

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
