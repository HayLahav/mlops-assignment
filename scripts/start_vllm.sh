#!/usr/bin/env bash
#
# vLLM launch config for Qwen3-30B-A3B-Instruct on 1× H100 80GB.
#
# Workload profile:
#   - 1.5–3K token prompts (DB schema + question)
#   - Short structured outputs (SQL, ~100–300 tokens)
#   - ~2–3 dependent LLM calls per user request (generate → verify → revise)
#
# SLO target: P95 end-to-end agent latency < 5s, 10+ RPS.
#
# Flag rationale (also in REPORT.md):
#
#   --dtype bfloat16
#       H100 has native BF16 tensor cores. No precision loss vs float32 for inference.
#
#   --gpu-memory-utilization 0.92
#       Use 92% of the 80 GB HBM. Leaves ~6 GB for CUDA kernels, PyTorch allocator
#       overhead, and activation buffers. Going higher risks OOM on long-tail requests.
#
#   --max-model-len 8192
#       The model supports 32K+ context but our prompts cap at ~3K tokens + short output.
#       Capping at 8192 dramatically reduces the KV cache block pool needed, which means
#       more concurrent sequences fit in memory.
#
#   --max-num-seqs 64
#       Caps the scheduler's concurrency window. MoE compute per token is cheap but KV
#       cache pressure grows with concurrency — 64 kept cache utilisation healthy without
#       evictions under the observed workload.
#
#   --enable-chunked-prefill
#       Without this, a 3K-token prefill monopolises the GPU for that entire step, adding
#       hundreds of milliseconds of head-of-line blocking to every in-flight decode request.
#       Chunked prefill interleaves prefill chunks with decode, cutting TTFT variance.
#
#   --enable-prefix-caching
#       Every request includes the DB schema in its prompt. Many requests target the same
#       DB, so the schema prefix is identical. Prefix caching reuses the computed KV for
#       that prefix on cache hits, reducing both TTFT and prompt-processing compute.
#
#   --trust-remote-code
#       Required for Qwen3's custom modelling code in the HuggingFace repo.
#
#   --disable-log-requests
#       Suppresses per-request access logs. Keeps stdout readable during load tests and
#       avoids contention on the log lock at high RPS.

set -euo pipefail

MODEL="${VLLM_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"

exec uv run python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.92 \
    --max-model-len 8192 \
    --max-num-seqs 64 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --disable-log-requests
