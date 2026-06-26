# GPU Kernels

GPU-kernel tasks adapted from [GPUMode](https://github.com/gpu-mode) and [KernelBench](https://github.com/ScalingIntelligence/KernelBench). Evolves a Triton kernel; the evaluator measures latency against the reference PyTorch implementation.

| Task | Op | What to evolve |
|------|-----|----------------|
| **trimul** | TriMul (triangular matrix multiplication) | `custom_kernel(...)` in `init_program.py` — either a CUDA kernel (`.cuda` reference) or a Triton implementation (`.triton` reference) |
| **cumsum** | Batched cumsum | `cumsum.triton` |
| **asymmetricmatmul** | Asymmetric matmul | `asymmetricmatmul.triton` |

Output must match the reference element-wise within the absolute and relative `TOLERANCE` specified in `server/tasks/{task}/task.yml`. The evaluated shapes are also declared in the same task.yml file. The score is computed as 1 / geomean(latency) over those declared shapes.

## Architecture

GPU tasks use a separate compiler-evaluation server instead of running candidates in the main evaluation subprocess. `evaluator.py` is a thin HTTP client that posts candidates to one of the servers in `server_info.yaml` (default `localhost:8000`).

### Launching the server

```bash
python datasets/gpukernel/server/launch_server.py \
  --port 8000 \
  --backend triton            # cuda | triton
```

`--triton-compiler-gpus` / `--triton-eval-gpus` split compile vs eval workloads across cards. See `--help`.

## Requirements

- NVIDIA GPU(s) with `nvcc` on PATH
- Python + `torch`

## Running a task

Once the server is up:

```bash
python main.py \
  --init-program datasets/gpukernel/trimul/init_program.py \
  --evaluator    datasets/gpukernel/trimul/evaluator.py \
  --instruction  datasets/gpukernel/trimul/level0_1_trimul.triton \
  --model        <your-model>
```

Kernel tasks pass the reference implementation (`.cuda` / `.triton`) as the instruction. The LLM sees what semantics to match, not a natural-language description.
