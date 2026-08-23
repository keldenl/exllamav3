# ExLLaMAV3 — KLin optimization fork

Downstream ExLLaMAV3 work focused on faster local inference while keeping the upstream project and its original README intact.

## Optimizations

- [EXL3 hot-vocabulary MTP drafting](https://github.com/keldenl/exllamav3/commit/5705f07b39671746af336bb004ad2e324410a654)
  - Reduces MTP draft-head work with aligned vocabulary selection, GPU-resident draft token chaining, and context-aware draft depth; measured about 21.9% higher mean decode throughput on an RTX 4060 Ti 16 GB.

Upstream project: [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3)
