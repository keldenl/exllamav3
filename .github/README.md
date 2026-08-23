# ExLLaMAV3 — Kelden’s downstream fork

This is a maintained downstream fork of [ExLLaMAV3](https://github.com/turboderp-org/exllamav3) with performance work aimed at practical local inference on consumer GPUs.

The upstream `dev` branch remains the clean upstream-development reference. The `downstream/dev` branch is the version that includes Kelden’s ongoing optimizations and improvements.

## Current improvements

- [Selected EXL3 hot vocabulary for MTP drafting](https://github.com/keldenl/exllamav3/commit/5705f07b39671746af336bb004ad2e324410a654)
  - Change: Adds an aligned, selected vocabulary head for the MTP draft model while retaining the full target vocabulary for verification.
  - Effect: Reduces the expensive draft output projection and measured about a 21.9% mean decode-throughput improvement in the balanced benchmark on an RTX 4060 Ti 16 GB.

- [GPU-resident draft embeddings and token chaining](https://github.com/keldenl/exllamav3/commit/5705f07b39671746af336bb004ad2e324410a654)
  - Change: Keeps the selected draft embeddings and intermediate proposed-token IDs on the GPU, limiting CPU readback to one read per draft block.
  - Effect: Removes avoidable host/device synchronization from the speculative-drafting path while keeping the full target model in charge of verification.

- [Context-aware MTP depth](https://github.com/keldenl/exllamav3/commit/5705f07b39671746af336bb004ad2e324410a654)
  - Change: Uses deeper MTP drafting at shallow context and automatically reduces the draft depth as live context grows.
  - Effect: Preserves the short-context speedup without forcing MTP-4 into the long-context regime where attention and KV-cache work dominate.

- [Aligned hot-vocabulary map builder and validation](https://github.com/keldenl/exllamav3/commit/5705f07b39671746af336bb004ad2e324410a654)
  - Change: Builds vocabulary maps on valid EXL3 Hadamard-group boundaries and adds numerical validation for selected-head choices.
  - Effect: Makes the optimization reproducible and avoids the invalid-logit behavior caused by arbitrary block selection.

## Relationship to upstream

The original project is maintained by [turboderp-org](https://github.com/turboderp-org/exllamav3). Changes in this fork may be proposed upstream when they are general enough, but `downstream/dev` is maintained as an independent, performance-focused branch.

For the original project documentation and complete feature list, see the [upstream README](https://github.com/turboderp-org/exllamav3/blob/master/README.md).
