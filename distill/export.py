#!/usr/bin/env python3
"""
Export training samples from Louter's SQLite database to JSONL files
for distillation fine-tuning.

Usage:
    python export.py                          # Export all unexported samples
    python export.py --task-type tool_call    # Export only tool_call samples
    python export.py --format sharegpt        # Export in ShareGPT format
    python export.py --db /path/to/louter.db  # Custom DB path
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path


def connect_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def export_samples(
    conn: sqlite3.Connection,
    task_type: str | None = None,
    format: str = "openai",
    limit: int = 0,
    mark_exported: bool = False,
) -> list[dict]:
    """Export training samples from the database."""
    query = "SELECT * FROM training_samples WHERE is_successful = 1 AND is_exported = 0"
    params = []

    if task_type:
        query += " AND task_type = ?"
        params.append(task_type)

    query += " ORDER BY created_at ASC"

    if limit > 0:
        query += " LIMIT ?"
        params.append(limit)

    cursor = conn.execute(query, params)
    rows = cursor.fetchall()

    samples = []
    exported_ids = []

    for row in rows:
        sample = convert_sample(dict(row), format)
        if sample:
            samples.append(sample)
            exported_ids.append(row["id"])

    if mark_exported and exported_ids:
        placeholders = ",".join(["?"] * len(exported_ids))
        conn.execute(
            f"UPDATE training_samples SET is_exported = 1 WHERE id IN ({placeholders})",
            exported_ids,
        )
        conn.commit()
        print(f"Marked {len(exported_ids)} samples as exported", file=sys.stderr)

    return samples


def convert_sample(row: dict, format: str) -> dict | None:
    """Convert a DB row to a training sample in the specified format."""
    try:
        messages = json.loads(row["request_messages"])
        response = json.loads(row["response_content"])
    except (json.JSONDecodeError, TypeError):
        return None

    if format == "openai":
        return convert_openai(messages, response, row)
    elif format == "sharegpt":
        return convert_sharegpt(messages, response, row)
    else:
        return convert_openai(messages, response, row)


def convert_openai(messages: list, response: dict, row: dict) -> dict:
    """OpenAI fine-tuning format: {"messages": [...]}"""
    conversation = list(messages)
    conversation.append(response)

    result = {"messages": conversation}

    # Include tools if present
    if row.get("request_tools"):
        try:
            tools = json.loads(row["request_tools"])
            if tools:
                result["tools"] = tools
        except (json.JSONDecodeError, TypeError):
            pass

    return result


def convert_sharegpt(messages: list, response: dict, row: dict) -> dict:
    """ShareGPT format: {"conversations": [{"from": "human", "value": "..."}]}"""
    role_map = {"system": "system", "user": "human", "assistant": "gpt", "tool": "tool"}

    conversations = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, list):
            # Multi-part content: extract text parts
            content = " ".join(
                p.get("text", "") for p in content if p.get("type") == "text"
            )
        from_role = role_map.get(role, role)
        conversations.append({"from": from_role, "value": content or ""})

    # Add response
    response_content = response.get("content", "")
    if response_content:
        conversations.append({"from": "gpt", "value": response_content})

    return {"conversations": conversations}


def print_stats(conn: sqlite3.Connection):
    """Print statistics about collected training samples."""
    cursor = conn.execute(
        """
        SELECT
            task_type,
            COUNT(*) as total,
            SUM(CASE WHEN is_successful = 1 THEN 1 ELSE 0 END) as successful,
            SUM(CASE WHEN has_tool_calls = 1 THEN 1 ELSE 0 END) as with_tools,
            SUM(CASE WHEN is_exported = 0 THEN 1 ELSE 0 END) as unexported,
            SUM(total_tokens) as total_tokens
        FROM training_samples
        GROUP BY task_type
        ORDER BY total DESC
    """
    )

    rows = cursor.fetchall()
    if not rows:
        print("No training samples found.", file=sys.stderr)
        return

    print("\n=== Training Sample Statistics ===\n", file=sys.stderr)
    print(
        f"{'Task Type':<15} {'Total':>8} {'Success':>8} {'Tools':>8} {'Unexport':>8} {'Tokens':>12}",
        file=sys.stderr,
    )
    print("-" * 65, file=sys.stderr)

    grand_total = 0
    for row in rows:
        row = dict(row)
        print(
            f"{row['task_type']:<15} {row['total']:>8} {row['successful']:>8} "
            f"{row['with_tools']:>8} {row['unexported']:>8} {row['total_tokens']:>12}",
            file=sys.stderr,
        )
        grand_total += row["total"]

    print("-" * 65, file=sys.stderr)
    print(f"{'TOTAL':<15} {grand_total:>8}", file=sys.stderr)
    print(file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Export Louter training samples")
    parser.add_argument(
        "--db", default="louter.db", help="Path to louter.db (default: louter.db)"
    )
    parser.add_argument(
        "--task-type",
        choices=["tool_call", "code", "math", "translation", "general"],
        help="Filter by task type",
    )
    parser.add_argument(
        "--format",
        choices=["openai", "sharegpt"],
        default="openai",
        help="Output format (default: openai)",
    )
    parser.add_argument(
        "--output", "-o", help="Output file (default: stdout)"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Max samples to export (0 = all)"
    )
    parser.add_argument(
        "--mark-exported",
        action="store_true",
        help="Mark exported samples so they won't be exported again",
    )
    parser.add_argument(
        "--stats", action="store_true", help="Print statistics and exit"
    )

    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}", file=sys.stderr)
        sys.exit(1)

    conn = connect_db(str(db_path))

    if args.stats:
        print_stats(conn)
        conn.close()
        return

    samples = export_samples(
        conn,
        task_type=args.task_type,
        format=args.format,
        limit=args.limit,
        mark_exported=args.mark_exported,
    )
    conn.close()

    if not samples:
        print("No samples to export.", file=sys.stderr)
        return

    # Write JSONL
    output = open(args.output, "w") if args.output else sys.stdout
    for sample in samples:
        output.write(json.dumps(sample, ensure_ascii=False) + "\n")

    if args.output:
        output.close()
        print(
            f"Exported {len(samples)} samples to {args.output}", file=sys.stderr
        )
    else:
        print(f"Exported {len(samples)} samples", file=sys.stderr)


if __name__ == "__main__":
    main()
