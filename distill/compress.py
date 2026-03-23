#!/usr/bin/env python3
"""
Training data compressor for Louter's distillation pipeline.

Compresses exported training samples by:
1. Deduplicating identical system prompts across samples
2. Truncating long tool results and system prompts
3. Removing Ring2 code blocks and verbose identity sections
4. Collapsing repeated context patterns

Usage:
    python compress.py input.jsonl -o compressed.jsonl
    python compress.py input.jsonl --max-system-tokens 200 --max-tool-result 500
    python compress.py input.jsonl --stats  # Show compression stats
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter


# Patterns to strip from system prompts
_RING2_CODE = re.compile(
    r"## Ring 2 Code\s*```[\s\S]*?```", re.MULTILINE
)
_PROTEA_STATE = re.compile(
    r"## Protea State[\s\S]*?(?=\n## |\Z)", re.MULTILINE
)
_VERBOSE_SECTIONS = re.compile(
    r"## (Cognitive Style|Communication|Autonomy|Response Style|Work Context)"
    r"[\s\S]*?(?=\n## |\Z)",
    re.MULTILINE,
)
# OpenClaw bootstrap patterns (very large, repetitive)
_OPENCLAW_BOOTSTRAP = re.compile(
    r"You are an AI assistant with access to.*?(?=\n\n|\Z)",
    re.DOTALL,
)


def compress_system_prompt(text: str, max_chars: int = 800) -> str:
    """Compress a system prompt, removing verbose/repetitive sections."""
    compressed = text

    # Remove code blocks
    compressed = _RING2_CODE.sub("[code omitted]\n", compressed)
    compressed = _PROTEA_STATE.sub("", compressed)
    compressed = _VERBOSE_SECTIONS.sub("", compressed)

    # Collapse whitespace
    compressed = re.sub(r"\n{3,}", "\n\n", compressed).strip()

    if len(compressed) > max_chars:
        compressed = compressed[:max_chars] + "\n...(compressed)"

    return compressed


def truncate_content(content, max_chars: int = 1000) -> str:
    """Truncate content (string or complex object) to max_chars."""
    if isinstance(content, str):
        if len(content) > max_chars:
            return content[:max_chars] + "...(truncated)"
        return content
    elif isinstance(content, list):
        # Multi-part content — extract text
        text_parts = [
            p.get("text", "") for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        ]
        text = " ".join(text_parts)
        if len(text) > max_chars:
            return text[:max_chars] + "...(truncated)"
        return text
    return str(content)[:max_chars]


def compress_sample(
    sample: dict,
    max_system_chars: int = 800,
    max_tool_result_chars: int = 500,
    max_user_chars: int = 2000,
    system_prompt_cache: dict | None = None,
) -> dict | None:
    """Compress a single training sample.

    Returns the compressed sample, or None if it should be skipped.
    """
    messages = sample.get("messages", [])
    if not messages:
        return None

    compressed_messages = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            # Compress system prompt
            if isinstance(content, str):
                compressed = compress_system_prompt(content, max_system_chars)

                # Deduplicate: if we've seen this system prompt before,
                # use a short reference instead
                if system_prompt_cache is not None:
                    prompt_hash = hashlib.md5(content.encode()).hexdigest()[:8]
                    if prompt_hash in system_prompt_cache:
                        system_prompt_cache[prompt_hash]["count"] += 1
                        # Use the compressed version
                        compressed = system_prompt_cache[prompt_hash]["compressed"]
                    else:
                        system_prompt_cache[prompt_hash] = {
                            "compressed": compressed,
                            "count": 1,
                            "original_len": len(content),
                        }

                compressed_messages.append({"role": "system", "content": compressed})
            else:
                compressed_messages.append(msg)

        elif role == "tool":
            # Truncate tool results
            content_str = truncate_content(content, max_tool_result_chars)
            new_msg = {"role": "tool", "content": content_str}
            if "tool_call_id" in msg:
                new_msg["tool_call_id"] = msg["tool_call_id"]
            compressed_messages.append(new_msg)

        elif role == "user":
            # Truncate very long user messages
            if isinstance(content, str) and len(content) > max_user_chars:
                compressed_messages.append({
                    "role": "user",
                    "content": content[:max_user_chars] + "...(truncated)",
                })
            else:
                compressed_messages.append(msg)

        elif role == "assistant":
            # Keep assistant messages as-is (training target)
            compressed_messages.append(msg)
        else:
            compressed_messages.append(msg)

    if len(compressed_messages) < 2:
        return None

    result = {"messages": compressed_messages}
    if "tools" in sample:
        result["tools"] = sample["tools"]

    return result


def compress_file(
    input_path: str,
    output_path: str | None = None,
    max_system_chars: int = 800,
    max_tool_result_chars: int = 500,
    max_user_chars: int = 2000,
    show_stats: bool = False,
) -> dict:
    """Compress an entire JSONL file.

    Returns statistics about the compression.
    """
    system_prompt_cache: dict = {}
    samples = []
    original_size = 0
    compressed_size = 0
    skipped = 0

    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            original_json = json.dumps(sample, ensure_ascii=False)
            original_size += len(original_json)

            compressed = compress_sample(
                sample,
                max_system_chars=max_system_chars,
                max_tool_result_chars=max_tool_result_chars,
                max_user_chars=max_user_chars,
                system_prompt_cache=system_prompt_cache,
            )

            if compressed:
                compressed_json = json.dumps(compressed, ensure_ascii=False)
                compressed_size += len(compressed_json)
                samples.append(compressed)
            else:
                skipped += 1

    # Write output
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    stats = {
        "input_samples": len(samples) + skipped,
        "output_samples": len(samples),
        "skipped": skipped,
        "original_size_kb": original_size / 1024,
        "compressed_size_kb": compressed_size / 1024,
        "compression_ratio": (
            (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
        ),
        "unique_system_prompts": len(system_prompt_cache),
        "system_prompt_stats": {
            h: {
                "count": v["count"],
                "original_len": v["original_len"],
                "compressed_len": len(v["compressed"]),
            }
            for h, v in system_prompt_cache.items()
        },
    }

    if show_stats:
        print("\n=== Compression Stats ===\n", file=sys.stderr)
        print(f"Input samples:  {stats['input_samples']}", file=sys.stderr)
        print(f"Output samples: {stats['output_samples']}", file=sys.stderr)
        print(f"Skipped:        {stats['skipped']}", file=sys.stderr)
        print(
            f"Original size:  {stats['original_size_kb']:.1f} KB", file=sys.stderr
        )
        print(
            f"Compressed:     {stats['compressed_size_kb']:.1f} KB", file=sys.stderr
        )
        print(
            f"Compression:    {stats['compression_ratio']:.1f}%", file=sys.stderr
        )
        print(
            f"\nUnique system prompts: {stats['unique_system_prompts']}",
            file=sys.stderr,
        )
        for h, v in stats["system_prompt_stats"].items():
            print(
                f"  [{h}] {v['original_len']} → {v['compressed_len']} chars "
                f"(seen {v['count']}x)",
                file=sys.stderr,
            )
        print(file=sys.stderr)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Compress Louter training data for distillation"
    )
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument(
        "--output", "-o", help="Output JSONL file (default: stdout)"
    )
    parser.add_argument(
        "--max-system-tokens",
        type=int,
        default=800,
        help="Max chars for system prompt (default: 800)",
    )
    parser.add_argument(
        "--max-tool-result",
        type=int,
        default=500,
        help="Max chars per tool result (default: 500)",
    )
    parser.add_argument(
        "--max-user",
        type=int,
        default=2000,
        help="Max chars for user messages (default: 2000)",
    )
    parser.add_argument(
        "--stats", action="store_true", help="Show compression statistics"
    )

    args = parser.parse_args()

    if args.output:
        stats = compress_file(
            args.input,
            args.output,
            max_system_chars=args.max_system_tokens,
            max_tool_result_chars=args.max_tool_result,
            max_user_chars=args.max_user,
            show_stats=True,
        )
    else:
        # Write to stdout
        stats = compress_file(
            args.input,
            None,
            max_system_chars=args.max_system_tokens,
            max_tool_result_chars=args.max_tool_result,
            max_user_chars=args.max_user,
            show_stats=args.stats,
        )
        # If no output file, write compressed samples to stdout
        if not args.output:
            # Re-read and compress to stdout
            system_prompt_cache: dict = {}
            with open(args.input) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        sample = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    compressed = compress_sample(
                        sample,
                        max_system_chars=args.max_system_tokens,
                        max_tool_result_chars=args.max_tool_result,
                        max_user_chars=args.max_user,
                        system_prompt_cache=system_prompt_cache,
                    )
                    if compressed:
                        print(json.dumps(compressed, ensure_ascii=False))


if __name__ == "__main__":
    main()
