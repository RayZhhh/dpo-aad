# Search with Trained Model

Run FunSearch using a DPO fine-tuned model to evaluate the quality of the trained adapter.

Before using this directory, complete the environment setup in the
[root README](../README.md) (clone AlgoDisco, install vLLM).
All commands must be run from the **`dpo-aad/` root**.

## Directory Structure

```
search_with_trained_model/
├── configs/
│   ├── funsearch_llama_trained.yaml   # Config template for the trained Llama model
│   └── funsearch_pangu_trained.yaml   # Config template for the trained Pangu model
├── serve_llama.sh                     # Start vLLM server with the Llama DPO adapter
├── serve_pangu.sh                     # Start vLLM server with the Pangu DPO adapter
├── run_search_llama.sh                # Launch parallel FunSearch runs (Llama)
├── run_search_pangu.sh                # Launch parallel FunSearch runs (Pangu)
└── stop_server.sh                     # Stop the running vLLM server
```

## Step 1: Start the vLLM Server

### Llama (port 8000)

```bash
bash search_with_trained_model/serve_llama.sh 8000 /path/to/llama_dpo_adapter meta-llama/Llama-3.1-8B-Instruct
```

### Pangu (port 8001)

```bash
bash search_with_trained_model/serve_pangu.sh 8001 /path/to/pangu_dpo_adapter /path/to/openPangu-Embedded-7B-V1.1
```

Wait until the server log prints `Application startup complete` before proceeding.

## Step 2: Run FunSearch

### Llama

```bash
bash search_with_trained_model/run_search_llama.sh [LORA_PATH] [BASE_MODEL] [PORT] [NUM_RUNS]
```

### Pangu

```bash
bash search_with_trained_model/run_search_pangu.sh [LORA_PATH] [BASE_MODEL] [PORT] [NUM_RUNS]
```

Both scripts launch `NUM_RUNS` (default: 3) independent FunSearch runs in the background.
Logs are written to `results/logs/funsearch_{llama,pangu}_trained_run{1,2,3}.out`.
Results are saved to `results/admissible_set/funsearch/{llama,pangu}_trained_run{1,2,3}/`.

To let the script start the server automatically, set `AUTO_START_SERVER=1`:

```bash
AUTO_START_SERVER=1 bash search_with_trained_model/run_search_llama.sh /path/to/llama_dpo_adapter
```

## Step 3: Stop the Server

```bash
bash search_with_trained_model/stop_server.sh
```

## Config Templates

Edit `configs/funsearch_llama_trained.yaml` (or the Pangu variant) to change:

- `method.template_program_path` / `method.task_description_path` — switch to a different task
- `method.max_samples` — total number of LLM samples per run
- `logger.logdir` — where results are saved

The `template_program_path` and `task_description_path` fields use paths relative to the
algodisco repo root, e.g. `task_examples/admissible_set/template_algo.txt`.

## SwanLab Logging

Set your API key before running:

```bash
export SWANLAB_API_KEY=your_key_here
```

Or replace `"${SWANLAB_API_KEY}"` directly in the config file.
