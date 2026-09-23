# ForgeRT

ForgeRT is a C++20/CUDA transformer inference runtime intended for latency-sensitive, small-batch inference. It is currently under development.

## Architecture

The intended layering is:

```text
CLI → Runtime → Model → Ops → CUDA Kernels → CUDA Infrastructure
```

Last updated: 2026-09-23
