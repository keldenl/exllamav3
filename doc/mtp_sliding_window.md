# Experimental Qwen MTP sliding-window attention

This branch can limit attention history only in the Qwen MTP drafter:

```bash
export EXL3_MTP_WINDOW=16384
```

Unset the variable or use `EXL3_MTP_WINDOW=full` for the original behavior. The
target model's attention modules are not modified.

Native MTP can also cap its draft count beyond a context threshold:

```bash
export EXL3_MTP_CONTEXT_DRAFT_THRESHOLD=24576
export EXL3_MTP_LONG_CONTEXT_DRAFT_TOKENS=3
```

Both variables must be set together. These controls are applied in
`iterate_draftmodel_mtp_gen()`, not the generic draft-model path.

## Cycle profiling

Set `EXL3_MTP_PROFILE=1` to print one `MTP_CYCLE` record per verification cycle.
It includes context length, draft/verification q lengths, proposed and accepted
tokens, CUDA-event timings for draft, target verification, sampling, repair and
the complete cycle, plus sampling wall time and rollback CPU time. Profiling
synchronizes once per cycle and is intended for diagnosis, not headline throughput
measurement.

## RTX 4060 Ti experiment

Model: Qwen3.8-27B EXL3 3.00 bpw, 64K FP8 MTP hot vocabulary, Q4 target and
draft KV, fixed MTP-3. Each case excluded initial graph construction/recapture and
measured 63-64 steady-state output tokens from an exact-length source prompt.

| Context | 2K | 4K | 8K | 16K | Full |
|---:|---:|---:|---:|---:|---:|
| 4K | 44.57 | 47.25 | 45.16 | 43.53 | 45.30 |
| 32K | 48.00 | 46.20 | 47.17 | 48.47 | 38.10* |
| 64K | 41.86 | 41.87 | 39.64 | 41.46 | 39.57 |
| 100K | 30.50 | 30.57 | 30.73 | 31.97 | 30.56 |

Values are decode tokens/second. `*` The full-attention 32K run diverged to a
different greedy completion and lower-acceptance trajectory, so it is not a clean
paired speed comparison. Fresh-process repeats confirmed that this stack is not
bitwise deterministic across speculative trajectories.

The end-to-end TPS differences above are trajectory measurements, not evidence of
an intrinsic sliding-window kernel speedup. They are primarily explained by how many
drafts were accepted and therefore how many target verification rounds were needed.
The normalized steady-state cycle costs are nearly flat:

| Context | 2K | 4K | 8K | 16K | Full |
|---:|---:|---:|---:|---:|---:|
| 64K ms/verification round | 80.46 | 80.45 | 80.73 | 81.25 | 80.86 |
| 64K verification rounds | 19 | 19 | 20 | 19 | 20 |
| 100K ms/verification round | 91.22 | 91.02 | 90.55 | 90.98 | 91.04 |
| 100K verification rounds | 23 | 23 | 23 | 22 | 23 |

Sliding-window MTP did not demonstrate a significant intrinsic cycle-latency
reduction through 100K context on this configuration. Observed end-to-end TPS
differences were primarily explained by variation in speculative acceptance and
verification-round count. At 100K, for example, 16K is about 90.98 ms/round versus
91.04 ms/round for the full control (2.0016 s / 22 and 2.0940 s / 23 respectively;
the table's full value uses the measured 91.04 ms/round from the logged run). The
feature may still become useful at substantially longer contexts or with a more
expensive/wider draft model.

A profiled 64K run showed steady-state MTP draft cost around 8 ms per cycle versus
roughly 70-80+ ms for target verification. Profiling also showed substantial
run-to-run variation in target verification timing between otherwise comparable
runs. The target model is untouched by `EXL3_MTP_WINDOW`, so that variation should
be treated as runtime/profiling noise unless independently reproduced, not attributed
to the sliding window.

This branch is an archived experiment. Keep the feature here for possible
200K-262K contexts, wider MTP drafting, or a different hardware/configuration
regime; do not merge or promote it to the main optimization branch based on these
results.

Reproduce with `tools/bench_mtp_window.py`. The raw run log is kept outside the
repository in the sibling experiments directory as `mtp-window-sweep.log`.
