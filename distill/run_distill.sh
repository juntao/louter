#!/bin/bash
# Louter Distillation Pipeline — End-to-end: export → train → merge → deploy to Ollama
#
# Usage:
#   ./run_distill.sh                    # Full pipeline with defaults
#   ./run_distill.sh --export-only      # Just export data
#   ./run_distill.sh --train-only       # Just train (data must exist)
#   ./run_distill.sh --deploy-only      # Just merge and deploy to Ollama
#
# Environment:
#   LOUTER_DB       Path to louter.db (default: ../louter.db)
#   BASE_MODEL      Base model for fine-tuning (default: Qwen/Qwen2.5-3B-Instruct)
#   OLLAMA_MODEL    Name for the Ollama model (default: louter-distilled)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOUTER_DB="${LOUTER_DB:-${SCRIPT_DIR}/../louter.db}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
OLLAMA_MODEL="${OLLAMA_MODEL:-louter-distilled}"
OUTPUT_DIR="${SCRIPT_DIR}/output"
MERGED_DIR="${SCRIPT_DIR}/merged_model"
DATA_FILE="${OUTPUT_DIR}/training_data.jsonl"

export_only=false
train_only=false
deploy_only=false

for arg in "$@"; do
    case $arg in
        --export-only) export_only=true ;;
        --train-only) train_only=true ;;
        --deploy-only) deploy_only=true ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

# Step 1: Export training data
if [ "$train_only" = false ] && [ "$deploy_only" = false ]; then
    echo "=== Step 1: Exporting training data ==="
    echo "Database: $LOUTER_DB"

    python3 "$SCRIPT_DIR/export.py" \
        --db "$LOUTER_DB" \
        --output "$DATA_FILE" \
        --mark-exported \
        --format openai

    # Print stats
    python3 "$SCRIPT_DIR/export.py" --db "$LOUTER_DB" --stats

    sample_count=$(wc -l < "$DATA_FILE" 2>/dev/null || echo "0")
    echo "Exported $sample_count samples to $DATA_FILE"

    if [ "$sample_count" -lt 10 ]; then
        echo "Warning: Very few samples ($sample_count). Consider collecting more data before training."
        if [ "$export_only" = true ]; then exit 0; fi
        echo "Continuing anyway..."
    fi

    # Compress training data
    echo ""
    echo "=== Step 1b: Compressing training data ==="
    COMPRESSED_FILE="${OUTPUT_DIR}/training_data_compressed.jsonl"
    python3 "$SCRIPT_DIR/compress.py" "$DATA_FILE" -o "$COMPRESSED_FILE" --stats
    compressed_count=$(wc -l < "$COMPRESSED_FILE" 2>/dev/null || echo "0")
    echo "Compressed $compressed_count samples to $COMPRESSED_FILE"
    # Use compressed data for training
    DATA_FILE="$COMPRESSED_FILE"

    if [ "$export_only" = true ]; then
        echo "Export complete. Data saved to: $DATA_FILE"
        exit 0
    fi
fi

# Step 2: Train LoRA adapter
if [ "$deploy_only" = false ]; then
    echo ""
    echo "=== Step 2: Training LoRA adapter ==="
    echo "Base model: $BASE_MODEL"
    echo "Training data: $DATA_FILE"

    if [ ! -f "$DATA_FILE" ]; then
        echo "Error: Training data not found at $DATA_FILE"
        echo "Run with --export-only first, or without flags for full pipeline."
        exit 1
    fi

    python3 "$SCRIPT_DIR/train.py" \
        --data "$DATA_FILE" \
        --base-model "$BASE_MODEL" \
        --output-dir "$OUTPUT_DIR" \
        --epochs 5 \
        --batch-size 8 \
        --gradient-accumulation 2 \
        --lr 5e-4 \
        --lora-r 8 \
        --lora-alpha 16

    echo "Training complete. Adapter saved to: $OUTPUT_DIR"

    if [ "$train_only" = true ]; then
        exit 0
    fi
fi

# Step 3: Merge, convert to GGUF, and deploy to Ollama
echo ""
echo "=== Step 3: Merging and converting to GGUF ==="

python3 "$SCRIPT_DIR/train.py" \
    --merge \
    --base-model "$BASE_MODEL" \
    --adapter-path "$OUTPUT_DIR" \
    --output-dir "$MERGED_DIR"

# Check if GGUF conversion succeeded (Modelfile will exist if it did)
if [ -f "$MERGED_DIR/Modelfile" ]; then
    # Deploy to Ollama
    if command -v ollama &> /dev/null; then
        echo ""
        echo "Deploying to Ollama as '$OLLAMA_MODEL'..."
        ollama create "$OLLAMA_MODEL" -f "$MERGED_DIR/Modelfile"
        echo ""
        echo "=== Deployment complete! ==="
        echo "Model: $OLLAMA_MODEL"
        echo ""
        echo "To use with Louter, add to louter.toml:"
        echo ""
        echo "  [hybrid]"
        echo "  enabled = true"
        echo "  local_provider = \"ollama\""
        echo "  local_model = \"$OLLAMA_MODEL\""
        echo "  cloud_provider = \"anthropic\""
        echo "  cloud_model = \"claude-sonnet-4-20250514\""
        echo "  min_local_success_rate = 0.7"
        echo "  min_samples = 20"
        echo "  fallback_enabled = true"
        echo "  local_task_types = [\"tool_call\", \"code\", \"general\"]"
    else
        echo ""
        echo "Ollama not found. Install Ollama first:"
        echo "  curl -fsSL https://ollama.com/install.sh | sh"
        echo ""
        echo "Then import the model:"
        echo "  ollama create $OLLAMA_MODEL -f $MERGED_DIR/Modelfile"
    fi
else
    echo ""
    echo "=== GGUF conversion was skipped ==="
    echo "Merged model (safetensors) saved at: $MERGED_DIR"
    echo ""
    echo "Ollama requires GGUF format for Qwen models. To convert manually:"
    echo "  git clone https://github.com/ggerganov/llama.cpp"
    echo "  pip install -r llama.cpp/requirements.txt"
    echo "  python llama.cpp/convert_hf_to_gguf.py $MERGED_DIR --outfile $MERGED_DIR/model-q4_k_m.gguf --outtype q4_k_m"
    echo ""
    echo "Then create an Ollama model:"
    echo "  echo 'FROM $MERGED_DIR/model-q4_k_m.gguf' > $MERGED_DIR/Modelfile"
    echo "  ollama create $OLLAMA_MODEL -f $MERGED_DIR/Modelfile"
fi
