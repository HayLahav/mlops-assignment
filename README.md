# LLM Inference + Observability

Text-to-SQL agent running **Qwen3-30B-A3B-Instruct-2507** on a single H100 80 GB, with full serving observability and an offline eval harness against BIRD-bench.

See `REPORT.md` for the full analysis, configuration rationale, SLO diagnosis, and results.

---

## Architecture

```
User question
     |
     v
Agent server (FastAPI, port 8001)
     |
     +- generate_sql --> vLLM (port 8000)
     +- execute      --> SQLite (BIRD-bench DBs)
     +- verify       --> vLLM
     +- revise       --> vLLM   (if verify fails, up to 3 iterations)

Observability
  vLLM /metrics --> Prometheus (port 9090) --> Grafana (port 3000)
  LangGraph traces --> Langfuse (port 3001)
```

## Stack

| Component | Version | Role |
|---|---|---|
| vLLM | 0.10.2 | OpenAI-compatible inference server |
| LangGraph | latest | Agent graph: generate -> execute -> verify -> revise |
| Langfuse | 4.7 | Trace waterfall + per-request metadata tags |
| Prometheus + Grafana | - | Latency / throughput / KV cache dashboard |
| BIRD-bench | dev subset | 30-question SQL eval set |

## Setup

```bash
git clone https://github.com/HayLahav/mlops-assignment.git
cd mlops-assignment
uv sync
cp .env.example .env          # add Langfuse keys
uv run python scripts/load_data.py
docker compose up -d
bash scripts/start_vllm.sh    # requires H100
uv run uvicorn agent.server:app --host 0.0.0.0 --port 8001
```

Forward ports over SSH to reach UIs from your laptop:
```bash
ssh -L 3000:localhost:3000 -L 9090:localhost:9090 \
    -L 3001:localhost:3001 -L 8000:localhost:8000 \
    -L 8001:localhost:8001 user@<vm-host>
```

## Usage

```bash
# Single query
curl -X POST http://localhost:8001/answer \
  -H "Content-Type: application/json" \
  -d '{"question": "List Ajax superpowers.", "db": "superhero"}'

# 30-question baseline eval
uv run python evals/run_eval.py --out results/eval_baseline.json

# Load test
uv run python load_test/driver.py --rps 2 --duration 300
```

## Results

| Metric | Value |
|---|---|
| Eval pass rate - iter 0 (generate only) | 26.7% |
| Eval pass rate - iter 1 (after first revise) | 30.0% |
| Average iterations per question | 1.6 |
| P95 latency @ 2 RPS | 4.0 s |
| Max sustainable RPS with P95 < 5 s | ~2 RPS |
