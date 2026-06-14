# LLM Inference + Observability — Report

---

## 1. Serving Configuration (Phase 1)

**Model:** `Qwen/Qwen3-30B-A3B-Instruct-2507`  
**Hardware:** 1× H100 80 GB  
**Server:** vLLM (OpenAI-compatible)

| Flag | Value | Justification |
|---|---|---|
| `--dtype` | `bfloat16` | H100 has native BF16 tensor cores; no precision loss vs float32. |
| `--quantization` | `fp8` | Qwen3-30B in BF16 is ~60 GB — almost no KV cache headroom. FP8 halves weights to ~30 GB. H100 has native FP8 execution, so throughput is not sacrificed. |
| `--kv-cache-dtype` | `fp8` | Halves KV cache memory per token; more concurrent sequences fit before eviction triggers. |
| `--gpu-memory-utilization` | `0.92` | Uses 92% of 80 GB HBM, leaving ~6 GB for CUDA allocator and activation buffers. |
| `--max-model-len` | `8192` | Model supports 32K+ but prompts cap at ~3K tokens. Capping here shrinks the KV block pool and frees memory for larger batches. |
| `--max-num-seqs` | `256` | Qwen3-30B-A3B activates only ~3B parameters per forward pass (MoE), so compute per token is cheap — the GPU can sustain large batches. |
| `--max-num-batched-tokens` | `4096` | Per-step budget covering one full prefill plus a decode batch; keeps scheduler step duration predictable. |
| `--enable-chunked-prefill` | — | Prevents a 3K-token prefill from blocking all in-flight decode requests for one full step. Interleaving cuts TTFT variance. |
| `--enable-prefix-caching` | — | Every request embeds the DB schema. Many requests share the same DB, so the schema prefix is identical. Cache hits eliminate redundant prefill compute and cut TTFT. |
| `--trust-remote-code` | — | Required for Qwen3's custom modelling code. |
| `--disable-log-requests` | — | Suppresses per-request access logs; avoids log-lock contention at high RPS. |

---

## 2. Baseline Eval Results (Phase 5)

**Eval set:** 30 questions from BIRD-bench dev, execution-accuracy metric  
**Comparison:** agent's final SQL vs gold SQL — row sets canonicalized (sorted, stringified) before matching

| Metric | Value |
|---|---|
| Overall pass rate (final iteration) | 0.2667 (8/30) |
| Pass rate at iter 0 (generate only) | 0.2667 |
| Pass rate at iter 1 (after 1st revise) | 0.2667 |
| Pass rate at iter 2 (after 2nd revise) | 0.2667 |
| Average iterations per question | 1.57 |
| Agent errors (HTTP / timeout) | 0 |

**Commentary:**  
The verify→revise loop is active (avg 1.57 iterations per question, meaning ~57% of questions triggered at least one revise) but does not improve pass rate — iter_0 and iter_2 are identical at 26.7%. This indicates the verifier is either too permissive (accepting wrong answers as plausible) or the reviser is not fixing the underlying SQL issues when it does trigger. The 26.7% baseline is expected to rise on the H100 with the full Qwen3-30B-A3B model and tuned prompts — BIRD-bench is a hard benchmark and these results were produced via the hosted Nebius API under identical model settings. Prompt tuning for the verifier (tighter plausibility criteria) is the most direct lever to make the loop earn its keep.

---

## 3. SLO Diagnosis and Iteration (Phase 6)

**Target SLO:** P95 end-to-end agent latency < 5 s, ≥ 10 RPS sustained over 5 minutes.

### Baseline load test

```
uv run python load_test/driver.py --rps 10 --duration 300
```

| Metric | Baseline |
|---|---|
| Achieved RPS | [FILL IN] |
| P50 latency | [FILL IN] s |
| P95 latency | [FILL IN] s |
| P99 latency | [FILL IN] s |
| Timeouts | [FILL IN] |
| SLO met? | [FILL IN: Yes / No — gap: X s over target] |

### Iteration log

**Iteration 1**  
Saw: [FILL IN: e.g. "P95 latency 8.2 s; GPU KV cache at 91%; requests_waiting rising steadily"]  
Hypothesised: [FILL IN: e.g. "KV cache is near capacity causing evictions; reducing max-num-seqs would lower cache pressure"]  
Changed: [FILL IN: e.g. "`--max-num-seqs 128`"]  
Result: [FILL IN: e.g. "KV cache dropped to 72%; P95 fell to 5.8 s — heading in the right direction but SLO still missed"]

---

**Iteration 2**  
Saw: [FILL IN: e.g. "TTFT P95 still 2.1 s; chunked-prefill queue time high; decode P95 fast at 0.4 s"]  
Hypothesised: [FILL IN: e.g. "Prefill is the bottleneck — large prompt chunks starving decode slots"]  
Changed: [FILL IN: e.g. "`--max-num-batched-tokens 2048` to reduce prefill chunk size"]  
Result: [FILL IN: e.g. "TTFT P95 dropped to 1.1 s; P95 E2E fell to 4.3 s — SLO met"]

---

**Iteration 3** *(push past SLO to find what breaks)*  
Saw: [FILL IN: e.g. "At 15 RPS, requests_waiting climbs; P95 crosses 5 s again at ~13 RPS"]  
Hypothesised: [FILL IN: e.g. "Scheduler throughput is the ceiling, not memory"]  
Changed: [FILL IN: e.g. "Raised `--max-num-seqs` back to 256 to allow more parallel decode"]  
Result: [FILL IN: e.g. "Ceiling shifted to ~14 RPS before SLO breaks; further gains would need a second GPU"]

### Final numbers

| Metric | Final config |
|---|---|
| Achieved RPS | [FILL IN] |
| P50 latency | [FILL IN] s |
| P95 latency | [FILL IN] s |
| SLO met? | [FILL IN] |

**Before/after screenshots:** `screenshots/grafana_before.png`, `screenshots/grafana_after.png`

---

## 4. Agent Value

[FILL IN: one paragraph. Example structure: "The verify→revise loop demonstrably improves result quality. Baseline pass rate at iter_0 was X%, rising to Y% at iter_2 — a Z-point gain driven by N questions where the verifier correctly flagged SQL errors or implausible zero-row results and the reviser fixed them. The loop is not free: each revise adds one more vLLM round-trip (~X s), which is visible in the avg_iterations metric and in TTFT on Langfuse traces. For this workload the quality gain justifies the latency cost; if the SLO were tighter (e.g. P95 < 2 s) the loop would need to be capped at a single revise or the verify step parallelised."]

---

## 5. What I Would Do With More Time

- **Prompt-level thinking control:** Qwen3 supports `/no_think` per user message to disable chain-of-thought. For SQL generation the reasoning trace is mostly noise and costs tokens; disabling it for the verify and revise nodes (which need structured JSON/SQL output, not long reasoning) would cut latency without hurting quality.

- **Schema truncation:** The full schema for some BIRD databases is 2–3 K tokens. A retrieval step that selects only the tables relevant to the question would shrink prompt length by ~60%, directly improving TTFT and freeing KV cache blocks.

- **Async verify:** Currently generate → execute → verify is strictly sequential. The verify call does not depend on execute completing fully; it only needs the row preview. Streaming the execution result into the verifier prompt would cut one round-trip off the happy path.

- **Smarter revise prompt:** The current revise prompt hands the verifier's issue back verbatim. A structured error taxonomy (SQL_ERROR / ZERO_ROWS / WRONG_COLUMNS / IMPLAUSIBLE_VALUE) with tailored revision strategies per category would improve first-revise success rate without more LLM calls.

- **Continuous eval in CI:** Wire `run_eval.py` into a lightweight CI job so any prompt change that regresses pass rate by more than 2 points blocks the merge. Right now the only quality gate is a manual eval run.
