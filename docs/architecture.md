# Architecture

ForgeRT is organized around a one-directional dependency flow:

```text
CLI → Runtime → Model → Ops → CUDA Kernels → CUDA Infrastructure
```

- `src/core/` will hold GPU-independent data structures and metadata.
- `src/cuda/` will hold CUDA runtime infrastructure, including device resources, allocation, streams, contexts, and error handling.
- `src/kernels/` will hold individual CUDA kernels and their launch wrappers.
- `src/ops/` will hold higher-level transformer operations that may compose kernels and vendor-library calls.
- `src/model/` will hold model semantics and weight/configuration representation.
- `src/runtime/` will hold execution concerns such as KV cache, generation, and orchestration.
- `src/io/` will define model loading and tokenization boundaries.
- `src/cli/` will hold the eventual executable entry point.
- `include/forgert/` is reserved for the public API.
- `tests/` and `benchmarks/` are separated by abstraction level and purpose.
- `tools/` is reserved for development-time utilities; `docs/` for architecture and performance records.

Model semantics should remain separate from CUDA execution details. Runtime scheduling/execution concerns should remain separate from model semantics.

Lower-level CUDA and kernel infrastructure must not depend on runtime or model layers.
