"""Controlled Qwen MTP attention-window throughput sweep.

Loads target and MTP weights once, then runs exact token-length source prompts. Only
the MTP Attention.sliding_window value changes between cases; target attention is
never modified.
"""

import argparse
import hashlib
import time
from pathlib import Path

import torch

from exllamav3 import (
    ArgmaxSampler,
    Cache,
    CacheLayer_quant,
    Config,
    Generator,
    Job,
    Model,
    MTPHotVocabConfig,
    Tokenizer,
)


def parse_csv_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def set_mtp_window(mtp, window: int):
    for attn in mtp.attn_modules:
        if attn.sliding_window == window:
            continue
        attn.sliding_window = window
        # BC attention captures the window as a constexpr. Force its lazy rebuild;
        # target-model graph caches are deliberately untouched.
        attn.bc_attn.clear()
    mtp.mtp_window = window


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required = True)
    parser.add_argument("--source-dir", required = True)
    parser.add_argument("--hot-blocks", required = True)
    parser.add_argument("--contexts", type = parse_csv_ints, default = "4096,32768,65536,100000")
    parser.add_argument("--windows", type = parse_csv_ints, default = "2048,4096,8192,16384,-1")
    parser.add_argument("--tokens", type = int, default = 64)
    parser.add_argument("--cache", type = int, default = 106496)
    parser.add_argument("--draft", type = int, default = 3)
    args = parser.parse_args()

    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    mtp = Model.from_config(
        config,
        component = "mtp",
        mtp_hot_vocab_config = MTPHotVocabConfig(
            blocks_path = args.hot_blocks,
            embedding_dtype = "fp8",
        ),
    )
    cache = Cache(
        model, max_num_tokens = args.cache, layer_type = CacheLayer_quant,
        max_batch_size = 1, max_history = args.draft, k_bits = 4, v_bits = 4,
    )
    mtp_cache = Cache(
        mtp, max_num_tokens = args.cache, layer_type = CacheLayer_quant,
        max_batch_size = 1, k_bits = 4, v_bits = 4,
    )
    mtp.load(progressbar = True)
    model.load(progressbar = True)
    tokenizer = Tokenizer.from_config(config)

    source = ""
    for path in sorted(Path(args.source_dir).rglob("*.py")):
        source += f"\n\n# FILE: {path}\n" + path.read_text(encoding = "utf-8", errors = "ignore")
    source_ids = tokenizer.encode(source, add_bos = False)
    suffix_ids = tokenizer.encode(
        "\n\nImplement a robust stable merge sort in Python, then explain its invariants.",
        add_bos = False,
    )
    if source_ids.shape[-1] < max(args.contexts):
        raise ValueError(f"Source corpus has only {source_ids.shape[-1]} tokens")

    sampler = ArgmaxSampler()
    print("SWEEP_HEADER", {
        "contexts": args.contexts,
        "windows": args.windows,
        "tokens": args.tokens,
        "draft": args.draft,
    })
    for window in args.windows:
        set_mtp_window(mtp, window)

        for context in args.contexts:
            prefix_len = context - suffix_ids.shape[-1]
            if prefix_len <= 0:
                raise ValueError(f"Context {context} is shorter than the prompt suffix")
            input_ids = torch.cat((source_ids[:, :prefix_len], suffix_ids), dim = -1)

            generator = Generator(
                model = model,
                cache = cache,
                tokenizer = tokenizer,
                draft_model = mtp,
                draft_cache = mtp_cache,
                num_draft_tokens = args.draft,
                record_draft_stats = True,
            )
            job = Job(
                input_ids = input_ids,
                max_new_tokens = args.tokens + 12,
                sampler = sampler,
                seed = 12345,
                stop_conditions = [],
            )
            generator.enqueue(job)
            output = []
            measure_start = None
            measure_token_start = 0
            measure_stats_start = 0
            started = time.perf_counter()
            while generator.num_remaining_jobs():
                for result in generator.iterate():
                    text = result.get("text", "")
                    if text:
                        output.append(text)
                # The first few cycles absorb per-window/per-context graph compilation
                # and recapture. Synchronize once at the measurement boundary, then
                # report only steady-state cycles from the same job and cache state.
                if measure_start is None and job.new_tokens >= 8:
                    torch.cuda.synchronize()
                    measure_start = time.perf_counter()
                    measure_token_start = job.new_tokens
                    measure_stats_start = len(job.draft_stats)
            torch.cuda.synchronize()
            ended = time.perf_counter()
            measured_stats = job.draft_stats[measure_stats_start:]
            measured_tokens = job.new_tokens - measure_token_start
            rounds = len(measured_stats)
            accepted = sum(x[2] for x in measured_stats)
            proposed = sum(x[1] for x in measured_stats)
            result = {
                "context": context,
                "mtp_window": window,
                "tokens": measured_tokens,
                "decode_s": ended - (measure_start or started),
                "decode_tps": measured_tokens / max(ended - (measure_start or started), 1e-9),
                "accepted": accepted,
                "proposed": proposed,
                "acceptance": accepted / proposed if proposed else 0.0,
                "accepted_per_round": accepted / rounds if rounds else 0.0,
                "committed_per_round": (accepted + rounds) / rounds if rounds else 0.0,
                "rounds": rounds,
                "output_sha256": hashlib.sha256("".join(output).encode()).hexdigest(),
            }
            print("SWEEP_RESULT", result, flush = True)
            del generator, job


if __name__ == "__main__":
    main()
