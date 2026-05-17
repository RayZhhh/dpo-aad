# Create DPO dataset

Pipeline for generating DPO preference data via LLM-driven algorithm search.

**Python >= 3.11 is required.**

Before running anything here, complete the environment setup in the
[root README](../README.md) (clone AlgoDisco into the `dpo-aad/` root, install vLLM).
All commands below are run from the **`create_data/`** directory.

## Step 1: Start the vLLM Server

```bash
bash start_vllm_server.sh
```

This serves `meta-llama/Llama-3.1-8B-Instruct` on `http://localhost:8000/v1`. Wait until the log prints `Application startup complete` before proceeding.

Set your SwanLab API key if you want experiment tracking — either way works:

```bash
# Option 1: export as environment variable
export SWANLAB_API_KEY=your_key_here

# Option 2: fill it in directly in the config file
# Edit configs/admi_funsearch_run1.yaml (and run2/run3 variants), replace:
#   api_key: "${SWANLAB_API_KEY}"
# with:
#   api_key: "your_key_here"
```

## Step 2: Run FunSearch (3 independent runs)

Three runs are launched in parallel to collect diverse search trajectories.
Run from the **`create_data/`** directory:

```bash
cd create_data

# Admissible-set task
bash start_funsearch_admi.sh

# CVRP task
bash start_funsearch_cvrp.sh
```

Each script starts `run1`, `run2`, and `run3` in the background. Logs are written to:

```
results/logs/funsearch_run{1,2,3}.out          # admissible set
results/logs/funsearch_cvrp_run{1,2,3}.out     # CVRP
```

Search results (programs + scores) are saved to:

```
results/admisible_set/funsearch/run{1,2,3}/
results/cvrp/funsearch/run{1,2,3}/
```

Configs for each run:

| Task           | Run  | Config                                  |
|----------------|------|-----------------------------------------|
| Admissible set | run1 | `configs/admi_funsearch_run1.yaml`      |
| Admissible set | run2 | `configs/admi_funsearch_run2.yaml`      |
| Admissible set | run3 | `configs/admi_funsearch_run3.yaml`      |
| CVRP           | run1 | `configs/cvrp_funsearch_run1.yaml`      |
| CVRP           | run2 | `configs/cvrp_funsearch_run2.yaml`      |
| CVRP           | run3 | `configs/cvrp_funsearch_run3.yaml`      |

Each run independently searches up to `max_samples=10000`. The three runs use the same hyperparameters; diversity comes from stochastic LLM sampling.
