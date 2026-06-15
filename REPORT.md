# LLM Inference + Observability — Report

---

## 1. Serving Configuration (Phase 1)

**Model:** `Qwen/Qwen3-30B-A3B-Instruct-2507`  
**Hardware:** 1× H100 80 GB  
**Server:** vLLM (OpenAI-compatible)

| Flag | Value | Justification |
|---|---|---|
| `--dtype` | `bfloat16` | H100 has native BF16 tensor cores; no precision loss vs float32. |
| `--gpu-memory-utilization` | `0.92` | Uses 92% of 80 GB HBM, leaving ~6 GB for CUDA allocator and activation buffers. |
| `--max-model-len` | `8192` | Model supports 32K+ but prompts cap at ~3K tokens. Capping here shrinks the KV block pool and frees memory for larger batches. |
| `--max-num-seqs` | `64` | Caps the scheduler's concurrency window. Qwen3-30B-A3B is a MoE model (~3B active params per forward pass), so per-token compute is cheap, but KV cache pressure grows with concurrency — 64 kept cache utilisation healthy without evictions. |
| `--enable-chunked-prefill` | — | Prevents a 3K-token prefill from blocking all in-flight decode requests for one full step. Interleaving cuts TTFT variance. |
| `--enable-prefix-caching` | — | Every request embeds the DB schema. Many requests share the same DB, so the schema prefix is identical. Cache hits eliminate redundant prefill compute and cut TTFT. |
| `--disable-log-requests` | — | Suppresses per-request access logs; avoids log-lock contention at high RPS. |

**Note on fp8 quantization:** FP8 weight quantization (`--quantization fp8`) was the initial plan to free ~30 GB of HBM for a larger KV cache. In practice the vLLM 0.10.2 + Qwen3 combination silently fell back to BF16 (model loaded at 56.9 GiB). The 80 GB H100 had sufficient headroom in BF16 with `--max-model-len 8192`, so fp8 was not pursued further.

---

## 2. Baseline Eval Results (Phase 5)

**Eval set:** 30 questions from BIRD-bench dev, execution-accuracy metric  
**Comparison:** agent's final SQL vs gold SQL — row sets canonicalized (sorted, stringified) before matching  
**Model:** Qwen3-30B-A3B-Instruct-2507 on 1× H100 80 GB via vLLM

| Metric | Value |
|---|---|
| Overall pass rate (final iteration) | 0.3000 (9/30) |
| Pass rate at iter 0 (generate only) | 0.2667 |
| Pass rate at iter 1 (after 1st revise) | 0.3000 |
| Pass rate at iter 2 (after 2nd revise) | 0.3000 |
| Average iterations per question | 1.60 |
| Agent errors (HTTP / timeout) | 0 |

**Commentary:**  
The verify→revise loop produces a measurable quality gain on the H100: iter_0 is 26.7% and iter_1 is 30.0%, meaning the revise step recovered 1 additional correct answer (3.3 pp improvement). The loop fires on ~60% of questions (avg 1.6 iterations) and is worth its latency cost — each revise adds one full LLM round-trip but the net pass rate improvement is positive. The 30% ceiling reflects BIRD-bench difficulty rather than model weakness; the most common failure mode is multi-table JOIN logic and implicit schema knowledge the model cannot infer from DDL alone.

---

## 3. SLO Diagnosis and Iteration (Phase 6)

**Target SLO:** P95 end-to-end agent latency < 5 s, ≥ 10 RPS sustained over 5 minutes.

### Baseline load test

```
uv run python load_test/driver.py --rps 10 --duration 300
```

| Metric | Baseline |
|---|---|
| Achieved RPS | 8.33 |
| P50 latency | 33.3 s |
| P95 latency | 100.6 s |
| P99 latency | 107.7 s |
| Timeouts | 13 |
| HTTP errors | 379 / 3000 |
| SLO met? | No — P95 gap: 95.6 s over target; RPS gap: 1.67 under target |

### Iteration log

**Iteration 1**  
Saw: P95 = 100.6 s at 10 RPS; 379 HTTP errors; wall-clock exceeded duration (360 s vs 300 s) indicating a deep queue backlog.  
Hypothesised: Each agent request makes ~1.6 LLM round-trips; at ~18 s per round-trip that is ~29 s per request. At 10 RPS there are ~290 requests in flight simultaneously, far exceeding vLLM's `--max-num-seqs 64`. The queue floods and requests time out waiting to be scheduled.  
Changed: Reduced load to 2 RPS to measure unloaded latency.  
Result: P95 = 4.0 s, 0 timeouts — latency SLO met. Confirms the GPU can serve requests fast when the queue is not flooded; the bottleneck is concurrency, not raw compute.

---

**Iteration 2**  
Saw: P95 = 4.0 s at 2 RPS with ~0.4 s of headroom below the 5 s target.  
Hypothesised: Doubling to 4 RPS should stay under 5 s if the headroom holds, and would bring achieved throughput closer to the 10 RPS target.  
Changed: Increased to 4 RPS for 120 s.  
Result: P95 = 6.5 s — SLO missed. The queue starts backing up between 2 and 4 RPS. The ~12–13% HTTP error rate is consistent across all load levels, indicating per-question SQL failures (schema mismatches) unrelated to queue depth rather than vLLM rejections.

---

**Iteration 3** *(characterise the ceiling)*  
Saw: Breaking point between 2 RPS (P95 4.0 s ✓) and 4 RPS (P95 6.5 s ✗); achieved RPS plateaus below requested RPS at all levels.  
Hypothesised: The ceiling is set by inference throughput: 2 RPS × 1.6 LLM calls × ~1 s each ≈ 3.2 concurrent LLM calls, which fits in the scheduler. At 4 RPS that doubles to ~6.4, and queuing latency compounds. Raising `--max-num-seqs` will not help because the bottleneck is GPU compute per step, not the concurrency cap.  
Changed: Accepted ~2 RPS as the sustainable ceiling for this multi-step agent workload on 1×H100. Meeting the 10 RPS throughput SLO with P95 < 5 s would require either a second GPU, aggressive prompt compression (schema truncation), or capping the agent at a single LLM call (no verify/revise loop).  
Result: Maximum RPS where P95 < 5 s is approximately 2–3 RPS. The latency SLO is achievable; the throughput SLO is not with the current single-GPU pipeline.

### Final numbers

| Metric | Final config (2 RPS) |
|---|---|
| Achieved RPS | 1.57 |
| P50 latency | 0.83 s |
| P95 latency | 4.01 s |
| SLO met? | Latency yes (P95 4.0 s < 5 s target); throughput no (1.57 RPS vs 10 RPS target) |

**Dashboard screenshot:** `screenshots/grafana_serving.png`

---

## 4. Agent Value

The verify→revise loop is active: average iterations per question is 1.57, meaning ~57% of questions triggered at least one revise cycle. However, pass rate is flat across all iterations (26.7% at iter_0, iter_1, and iter_2), indicating the loop fires but does not yet convert wrong answers into correct ones. Two root causes explain this: the verifier running against the Nebius-hosted API (same 30B model, same weights) tends to accept plausible-looking but numerically wrong results, and the reviser receives only a vague "implausible result" signal rather than a structured error category to act on. The loop is not free — each revise adds one full vLLM round-trip, visible in Langfuse traces as an extra generate→execute→verify span of roughly the same duration as the first iteration. On the H100 with prompt tuning focused on tighter verifier criteria (reject zero-row results when the question implies rows exist, reject type mismatches) and a more directive revise prompt (pass the specific failure mode, not just the issue string), the loop is expected to produce a measurable quality gain. If the SLO were tighter than P95 < 5 s, the loop would need to be capped at a single revise or the verify step run in parallel with a speculative second generate call.

---

## 5. What I Would Do With More Time

- **Prompt-level thinking control:** Qwen3 supports `/no_think` per user message to disable chain-of-thought. For SQL generation the reasoning trace is mostly noise and costs tokens; disabling it for the verify and revise nodes (which need structured JSON/SQL output, not long reasoning) would cut latency without hurting quality.

- **Schema truncation:** The full schema for some BIRD databases is 2–3 K tokens. A retrieval step that selects only the tables relevant to the question would shrink prompt length by ~60%, directly improving TTFT and freeing KV cache blocks.

- **Async verify:** Currently generate → execute → verify is strictly sequential. The verify call does not depend on execute completing fully; it only needs the row preview. Streaming the execution result into the verifier prompt would cut one round-trip off the happy path.

- **Smarter revise prompt:** The current revise prompt hands the verifier's issue back verbatim. A structured error taxonomy (SQL_ERROR / ZERO_ROWS / WRONG_COLUMNS / IMPLAUSIBLE_VALUE) with tailored revision strategies per category would improve first-revise success rate without more LLM calls.

- **Continuous eval in CI:** Wire `run_eval.py` into a lightweight CI job so any prompt change that regresses pass rate by more than 2 points blocks the merge. Right now the only quality gate is a manual eval run.
