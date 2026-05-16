# Create DPO dataset

Pipeline for generating DPO preference data via LLM-driven algorithm search.

## Step 1: Environment Setup

**Python >= 3.11 is required.**

### Install AlgoDisco

**Option 1 — pip (recommended):**

```bash
pip install "algodisco[swanlab]"
```

**Option 2 — from source:**

```bash
git clone https://github.com/RayZhhh/algodisco.git
cd algodisco
pip install -e ".[swanlab]"
cd ..
```

> The `swanlab` extra installs the experiment logger used in the configs.

### Install vLLM

vLLM is used as the local LLM backend. Installing it also pulls in the `openai` package automatically, so no separate `pip install openai` is needed.

```bash
pip install vllm
```

## Step 2: Start the vLLM Server

```bash
bash start_vllm_server.sh
```

This serves `meta-llama/Llama-3.1-8B-Instruct` on `http://localhost:8000/v1`. Wait until the log prints `Application startup complete` before proceeding.

Set your SwanLab API key if you want experiment tracking — either way works:

```bash
# Option 1: export as environment variable
export SWANLAB_API_KEY=your_key_here

# Option 2: fill it in directly in the config file
# Edit configs/admi_funsearch.yaml (and run2/run3 variants), replace:
#   api_key: "${SWANLAB_API_KEY}"
# with:
#   api_key: "your_key_here"
```

## Step 3: Run FunSearch (3 independent runs)

Three runs are launched in parallel to collect diverse search trajectories:

```bash
cd create_data
bash start_funsearch.sh
```

This starts `run1`, `run2`, and `run3` in the background. Logs are written to:

```
results/logs/funsearch_run1.out
results/logs/funsearch_run2.out
results/logs/funsearch_run3.out
```

Search results (programs + scores) are saved to:

```
results/admisible_set/funsearch/run{1,2,3}/
```

Configs for each run:

| Run  | Config                             |
|------|------------------------------------|
| run1 | `configs/admi_funsearch_run1.yaml` |
| run2 | `configs/admi_funsearch_run2.yaml` |
| run3 | `configs/admi_funsearch_run3.yaml` |

Each run independently searches for admissible set algorithms up to `max_samples=10000`. The three runs use the same hyperparameters; diversity comes from stochastic LLM sampling.
