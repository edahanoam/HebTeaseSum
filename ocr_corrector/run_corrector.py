"""Hebrew OCR correction pipeline: load data, correct via API, cache, export CSV."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pandas as pd
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

from src.data_loader import HebrewDataLoader
from src.corrector import OCRCorrector


def normalize_cache_key(text: str) -> str:
    """Stable key for deduplication (strip surrounding whitespace)."""
    return (text or "").strip()


def load_cache(path: Path) -> Dict[str, str]:
    """Load correction cache from JSON."""
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            return {}
        return {normalize_cache_key(k): v for k, v in raw.items() if isinstance(k, str)}
    except json.JSONDecodeError:
        print("Warning: Cache file corrupted. Starting fresh.")
        return {}


def save_cache(cache: Dict[str, str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def get_unique_texts(df: pd.DataFrame) -> Set[str]:
    texts: Set[str] = set()
    if "summary" in df.columns:
        for x in df["summary"].dropna().astype(str).tolist():
            k = normalize_cache_key(x)
            if k:
                texts.add(k)
    if "text" in df.columns:
        for x in df["text"].dropna().astype(str).tolist():
            k = normalize_cache_key(x)
            if k:
                texts.add(k)
    return texts


def apply_corrections_to_dataframe(df: pd.DataFrame, cache: Dict[str, str]) -> pd.DataFrame:
    """Map corrected strings using normalized cache keys."""
    out = df.copy()

    def lookup(val: Any) -> Any:
        if pd.isna(val):
            return val
        s = normalize_cache_key(str(val))
        if not s:
            return val
        return cache.get(s, val)

    if "summary" in out.columns:
        out["corrected_summary"] = out["summary"].map(lookup)
    if "text" in out.columns:
        out["corrected_text"] = out["text"].map(lookup)
    return out


def _is_rate_limit_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    if "429" in msg or "rate" in msg or "too many requests" in msg:
        return True
    name = type(exc).__name__.lower()
    return "ratelimit" in name or "rate_limit" in name


async def correct_one_with_retry(
    corrector: OCRCorrector,
    text: str,
    use_hybrid: bool,
    use_strong_only: bool,
    max_retries: int = 6,
) -> str:
    """Call corrector with exponential backoff on rate limits."""
    delay = 2.0
    for attempt in range(max_retries):
        try:
            return await corrector.correct_text(
                text,
                use_hybrid=use_hybrid,
                use_strong_only=use_strong_only,
            )
        except Exception as e:
            if _is_rate_limit_error(e) and attempt < max_retries - 1:
                await asyncio.sleep(delay)
                delay = min(delay * 2, 120.0)
                continue
            print(f"Warning: correction failed after {attempt + 1} attempt(s): {e}")
            return text
    return text


async def process_batch(
    corrector: OCRCorrector,
    texts: List[str],
    pbar: tqdm,
    semaphore: asyncio.Semaphore,
    use_hybrid: bool,
    use_strong_only: bool,
) -> List[Tuple[str, str]]:
    """Process texts with bounded concurrency; failures keep original text."""

    async def run_one(t: str) -> Tuple[str, str]:
        async with semaphore:
            corrected = await correct_one_with_retry(
                corrector, t, use_hybrid, use_strong_only
            )
        return (t, corrected)

    results = await asyncio.gather(
        *[run_one(t) for t in texts],
        return_exceptions=True,
    )

    out: List[Tuple[str, str]] = []
    for t, res in zip(texts, results):
        if isinstance(res, Exception):
            print(f"Warning: task failed for one text: {res}")
            out.append((t, t))
        else:
            out.append(res)
    pbar.update(len(texts))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hebrew OCR correction pipeline (OpenAI / Gemini)."
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=Path("data/hebrew_summary_data.csv"),
        help="Input CSV path",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/processed_data.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=Path("data/correction_cache.json"),
        help="JSON cache for deduplication",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "gemini"],
        default="openai",
    )
    parser.add_argument(
        "--prompt-version",
        choices=["v1", "v2"],
        default=None,
        help="System prompt version (overrides OCR_PROMPT_VERSION if set)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--weak-only",
        action="store_true",
        help="Use weak model for full text (default if no other mode flag)",
    )
    mode.add_argument(
        "--hybrid",
        action="store_true",
        help="Strong model for head, weak for body",
    )
    mode.add_argument(
        "--strong-only",
        action="store_true",
        help="Strong model for entire text (no split)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Process only first N rows after load (smoke test)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Number of texts to schedule per inner batch",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Max parallel API calls (semaphore)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Save cache + output CSV after this many newly corrected texts",
    )
    return parser.parse_args()


def resolve_mode(args: argparse.Namespace) -> Tuple[bool, bool]:
    """Return (use_hybrid, use_strong_only). Strong-only overrides hybrid."""
    if args.strong_only:
        if args.hybrid:
            print("Note: --strong-only takes precedence over --hybrid.")
        return (False, True)
    if args.hybrid:
        return (True, False)
    # default: weak-only unless user passed nothing — treat as weak-only
    return (False, False)


async def async_main() -> int:
    load_dotenv()
    args = parse_args()
    use_hybrid, use_strong_only = resolve_mode(args)

    print("--- Starting Hebrew OCR Correction Pipeline ---")

    loader = HebrewDataLoader()
    try:
        df = loader.load_raw_data(str(args.data_file))
    except FileNotFoundError:
        print(f"Error: Could not find {args.data_file}.")
        return 1
    except ValueError as e:
        print(f"Error loading data: {e}")
        return 1

    if args.sample is not None:
        if args.sample <= 0:
            print("Error: --sample must be a positive integer.")
            return 1
        print(f"Sample mode: processing first {args.sample} rows only.")
        df = df.head(args.sample)

    print(f"Loaded data: {df.shape[0]} rows.")

    prompt_kw: Dict[str, Any] = {}
    if args.prompt_version is not None:
        prompt_kw["prompt_version"] = args.prompt_version

    try:
        corrector = OCRCorrector(provider=args.provider, **prompt_kw)
    except ValueError as e:
        print(f"Error initializing corrector: {e}")
        return 1

    print(f"Provider: {corrector.provider}")
    prompt_label = "long prompt" if corrector.prompt_version == "v1" else "short prompt"
    print(f"Prompt version: {corrector.prompt_version} ({prompt_label})")
    if use_strong_only:
        print(f"Model mode: strong-only ({corrector.strong_model})")
    elif use_hybrid:
        print(
            "Model mode: hybrid "
            f"(strong={corrector.strong_model}, weak={corrector.weak_model})"
        )
    else:
        print(f"Model mode: weak-only ({corrector.weak_model})")
    print(
        f"Concurrency: max_concurrent={args.max_concurrent}, batch_size={args.batch_size}, "
        f"checkpoint every {args.checkpoint_every} new corrections"
    )

    correction_cache = load_cache(args.cache_file)
    print(f"Loaded {len(correction_cache)} cached corrections.")

    unique_texts = get_unique_texts(df)
    to_process = [t for t in unique_texts if t not in correction_cache]

    print(f"Total unique texts: {len(unique_texts)}")
    print(f"Remaining to process: {len(to_process)}")

    semaphore = asyncio.Semaphore(max(1, args.max_concurrent))
    processed_since_checkpoint = 0

    if to_process:
        with tqdm(total=len(to_process), desc="Correcting OCR") as pbar:
            for i in range(0, len(to_process), args.batch_size):
                batch = to_process[i : i + args.batch_size]
                batch_results = await process_batch(
                    corrector,
                    batch,
                    pbar,
                    semaphore,
                    use_hybrid=use_hybrid,
                    use_strong_only=use_strong_only,
                )
                for original, corrected in batch_results:
                    correction_cache[original] = corrected
                processed_since_checkpoint += len(batch_results)

                if processed_since_checkpoint >= args.checkpoint_every:
                    save_cache(correction_cache, args.cache_file)
                    df_ckpt = apply_corrections_to_dataframe(df, correction_cache)
                    args.output_file.parent.mkdir(parents=True, exist_ok=True)
                    df_ckpt.to_csv(
                        args.output_file, index=False, encoding="utf-8-sig"
                    )
                    print(
                        f"Checkpoint: saved cache and {args.output_file} "
                        f"({processed_since_checkpoint} new corrections in this window)."
                    )
                    processed_since_checkpoint = 0

        save_cache(correction_cache, args.cache_file)
        print("Processing complete. Cache saved.")
    else:
        print("All texts already cached. Skipping API calls.")

    print("Applying corrections to DataFrame...")
    df_out = apply_corrections_to_dataframe(df, correction_cache)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.output_file, index=False, encoding="utf-8-sig")
    print(f"--- Done! Output written to {args.output_file.resolve()} ---")
    return 0


def main() -> None:
    sys.exit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
