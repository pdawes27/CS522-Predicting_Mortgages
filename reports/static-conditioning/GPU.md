# CUDA verification and matched repeat

The three-seed CPU comparison remains the primary architecture experiment; this is a separate seed-522 GPU repeat.

Hardware: NVIDIA GeForce RTX 5070; PyTorch 2.15.0+cu132; CUDA 13.2.
Peak allocated CUDA tensor memory: 0.083 GiB; total run wall time: 55.5 seconds.

Saved population fingerprints and training-fitted preprocessing match the CPU comparison exactly. Both variants used CUDA for model weights, minibatches, forward/backward operations, and updates. Data preparation remained on the CPU.

| Model | Payoff AP | Payoffs in top 1% | Macro-F1 | Log loss | Median GPU epoch (s) | Median CPU epoch (s) |
|---|---:|---:|---:|---:|---:|---:|
| transformer | 0.04392 | 604 | 0.3220 | 0.09544 | 3.8 | 32.6 |
| transformer_conditioned | 0.03840 | 522 | 0.3036 | 0.09446 | 3.4 | 35.2 |

Epoch timings include training and validation. CPU runs overlapped other CPU fits, so these measurements are descriptive rather than an isolated hardware benchmark. GPU and CPU floating-point paths can select different checkpoints and predictions; cross-device bitwise equivalence is not expected.

The GPU repeat also gives lower payoff AP to the conditioned variant. It does not reverse the primary experiment conclusion.

Use the `.venv-gpu` interpreter with `--device cuda`; a CPU-only PyTorch installation cannot use the GPU. The agent must execute GPU commands outside its restricted sandbox. [README GPU instructions](../../README.md#gpu-training).
