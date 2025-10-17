# Partition feedback log into a knowledge base (JSONL partitions).
# Usage examples:
#  python src/partition_feedback.py --input feedback.csv --out_dir knowledge_base/partitioned --date_col Date --min_chunk_words 50 --max_chunk_words 400
#
# The script expects feedback log columns like:
#  Date, OMSTenantId, PolicyLevel, PolicyID, PolicyPriority, GroupID, SettingName, SettingValue, FeedbackText
#
# It creates JSONL files under out_dir/<yyyy-mm-dd>/tenant=<OMSTenantId>/part-*.jsonl
# Each JSON object contains: id, tenant, policy_key (composite), setting_name, original_ts, chunk_text, chunk_index, source_file.

import argparse
import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Iterable, Tuple

SENT_SPLIT_RE = re.compile(r'(?<=[\.\?\!])\s+')

def chunk_text(text: str, min_words: int = 50, max_words: int = 400) -> List[str]:
    """
    Simple sentence-based chunker. Attempts to create chunks between min_words and max_words.
    If a sentence is longer than max_words, it will be split by spaces.
    """
    if not text:
        return []
    sentences = [s.strip() for s in SENT_SPLIT_RE.split(text) if s.strip()]
    chunks = []
    cur = []
    cur_words = 0
    for s in sentences:
        wcount = len(s.split())
        if cur_words + wcount <= max_words:
            cur.append(s)
            cur_words += wcount
        else:
            # flush current if it's long enough, otherwise try to include sentence anyway
            if cur and cur_words >= min_words:
                chunks.append(" ".join(cur))
                cur = [s]
                cur_words = wcount
            else:
                # forced split: join current and sentence even if > max
                cur.append(s)
                chunks.append(" ".join(cur))
                cur = []
                cur_words = 0
    if cur:
        chunks.append(" ".join(cur))
    # final pass: ensure no chunk is empty and split extremely large chunks by words
    fixed = []
    for c in chunks:
        wc = len(c.split())
        if wc <= max_words:
            fixed.append(c)
        else:
            words = c.split()
            i = 0
            while i < len(words):
                fixed.append(" ".join(words[i:i+max_words]))
                i += max_words
    return fixed

def normalize_date(ts_str: str) -> str:
    if not ts_str:
        return "unknown"
    # try common ISO parse first
    try:
        dt = datetime.fromisoformat(ts_str)
    except Exception:
        # fallback: try date only
        try:
            dt = datetime.strptime(ts_str, "%Y-%m-%d")
        except Exception:
            # as last resort, attempt many common formats or return "unknown"
            try:
                dt = datetime.strptime(ts_str, "%m/%d/%Y %H:%M:%S")
            except Exception:
                return "unknown"
    return dt.date().isoformat()

def read_rows_from_csv(path: Path) -> Iterable[Dict]:
    with path.open(newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            yield r

def read_rows_from_jsonl(path: Path) -> Iterable[Dict]:
    with path.open(encoding='utf-8') as fh:
        for line in fh:
            if not line.strip():
                continue
            yield json.loads(line)

def ensure_dir(p: Path):
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)

def write_jsonl(path: Path, objects: Iterable[Dict]):
    ensure_dir(path.parent)
    with path.open('a', encoding='utf-8') as fh:
        for o in objects:
            fh.write(json.dumps(o, ensure_ascii=False) + '\n')

def build_policy_key(row: Dict) -> str:
    # Composite key as requested: OMSTenantId|PolicyLevel|PolicyID|PolicyPriority|GroupID
    parts = [
        row.get("OMSTenantId",""),
        row.get("PolicyLevel",""),
        row.get("PolicyID",""),
        row.get("PolicyPriority",""),
        row.get("GroupID","")
    ]
    return "|".join(str(p) for p in parts)

def partition_file(input_path: Path, out_dir: Path, date_col: str, min_chunk_words: int, max_chunk_words: int, feedback_col: str):
    # Determine reader by extension
    ext = input_path.suffix.lower()
    if ext in ('.csv',):
        rows = read_rows_from_csv(input_path)
    else:
        # assume JSONL
        rows = read_rows_from_jsonl(input_path)

    written_files = []
    total_chunks = 0
    for row in rows:
        ts = row.get(date_col) or row.get("Date") or ""
        date = normalize_date(ts)
        tenant = row.get("OMSTenantId") or "unknown"
        policy_key = build_policy_key(row)
        setting_name = row.get("SettingName") or row.get("setting") or ""
        text = row.get(feedback_col) or row.get("FeedbackText") or row.get("Comment") or ""
        chunks = chunk_text(text, min_words=min_chunk_words, max_words=max_chunk_words)
        # if no chunks but there is a SettingValue or other info, still create a small record
        if not chunks:
            chunks = [text] if text else []
        if not chunks:
            continue
        # prepare output folder
        base_out = out_dir / date / f"tenant={tenant}"
        ensure_dir(base_out)
        out_file = base_out / f"part-{tenant}-{date}.jsonl"
        objs = []
        for i, c in enumerate(chunks):
            obj = {
                "id": f"{tenant}__{policy_key}__{setting_name}__{date}__{i}",
                "tenant": tenant,
                "policy_key": policy_key,
                "policy_level": row.get("PolicyLevel"),
                "policy_id": row.get("PolicyID"),
                "policy_priority": row.get("PolicyPriority"),
                "group_id": row.get("GroupID"),
                "setting_name": setting_name,
                "original_ts": ts,
                "chunk_index": i,
                "chunk_text": c,
                "source_file": str(input_path.name)
            }
            objs.append(obj)
        write_jsonl(out_file, objs)
        total_chunks += len(objs)
        written_files.append(str(out_file))
    return {"files_written": list(set(written_files)), "total_chunks": total_chunks}

def main():
    parser = argparse.ArgumentParser(description="Partition feedback logs into knowledge base JSONL.")
    parser.add_argument("--input", "-i", required=True, help="Input file (CSV or JSONL) with feedback records.")
    parser.add_argument("--out_dir", "-o", default="knowledge_base/partitioned", help="Output folder.")
    parser.add_argument("--date_col", default="Date", help="Name of the timestamp/date column.")
    parser.add_argument("--feedback_col", default="FeedbackText", help="Column containing user feedback text.")
    parser.add_argument("--min_chunk_words", type=int, default=50, help="Minimum words per chunk.")
    parser.add_argument("--max_chunk_words", type=int, default=400, help="Maximum words per chunk.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print("Input file not found:", input_path)
        return
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    result = partition_file(input_path, out_dir, args.date_col, args.min_chunk_words, args.max_chunk_words, args.feedback_col)
    print("Wrote:", len(result["files_written"]), "files; total chunks:", result["total_chunks"])
    for f in result["files_written"][:20]:
        print("  ", f)

if __name__ == "__main__":
    main()
