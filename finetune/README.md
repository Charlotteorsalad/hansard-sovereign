# QLoRA fine-tune — teaching a small model the citation format

**Motivation.** Small models don't reliably follow the strict output format
(`N. **Title**: Speaker (Constituency) summary [n].`), so the RAG currently
enforces it with Python post-processing (see the note in the main README). This
fine-tune **distills that format behaviour into a 1.5B model** so it produces the
format natively — shrinking the brittle post-processing and letting a fast,
fully-GPU-resident small model do the job.

It's a self-distillation: the **teacher is the existing pipeline** (real
retrieval + the 8B model + the same Python cleanup), so the training targets are
exactly what production already emits.

## Just want to run the app with the fine-tune available?

The trained model (`hansard-qwen`, q4_K_M, ~920 MB) is published as a GitHub
Release so you don't need to redo steps 1–3 below just to use the model
selector / Compare mode:

```bash
bash scripts/fetch_model.sh   # downloads + `ollama create`s hansard-qwen
```

The rest of this doc is the actual training pipeline, for reproducing or
retraining the fine-tune itself.

## Workflow

```
build_finetune_data.py   →  train_qlora.py (Colab T4)  →  Ollama  →  evaluate.py
   (local, CPU)               (GPU training)              (deploy)     (measure)
```

### 1. Build the dataset (local)
Uses the production pipeline as teacher; keeps only well-formatted answers.
```bash
CUDA_VISIBLE_DEVICES="" uv run python finetune/build_finetune_data.py 120
# → finetune/train.jsonl + finetune/val.jsonl  ({system, prompt, completion})
```
Slow (the 8B teacher spills to CPU) — it writes incrementally to
`examples.jsonl`, so progress survives interruption.

### 2. Train on Colab (free T4 — the 4 GB laptop can't train)
Open `train_qlora.py` in Colab (cells are `# %%`-delimited), upload
`train.jsonl` + `val.jsonl`, run top to bottom. It:
- loads Qwen2.5-1.5B in 4-bit, attaches LoRA adapters (QLoRA),
- trains ~3 epochs with loss on the answer only,
- exports a **q4_K_M GGUF** ready for Ollama.

Training on a cloud T4 and deploying the result on the constrained laptop is the
"across diverse hardware profiles" story in one artifact.

### 3. Deploy the adapter to Ollama (local)
Download `hansard-qwen-gguf/` from Colab, convert it, then import with a
Modelfile that carries a real chat template — see the **Gotcha** section below
for why a naive `FROM <gguf>` Modelfile doesn't work. `finetune/Modelfile` is
the actual working one, exported straight from the deployed model
(`ollama show hansard-qwen --modelfile`):
```bash
ollama create hansard-qwen -f finetune/Modelfile   # after fixing its FROM path
```

### 4. Measure the payoff
Compares native format-adherence (raw output, no Python cleanup) of the base vs
the fine-tuned model:
```bash
CUDA_VISIBLE_DEVICES="" uv run python finetune/evaluate.py qwen2.5:1.5b hansard-qwen
```
The base 1.5B rarely holds the `**Title**: … [n]` format on its own; the goal is
for the fine-tune to do it natively, so the RAG can lean on a smaller, faster,
fully-GPU-resident model with less Python scaffolding.

**Result** (14 held-out queries, raw output, no Python cleanup):

| model | native format-adherence |
| --- | --- |
| `qwen2.5:1.5b` (base) | 1/14 — **7%** |
| `hansard-qwen` (QLoRA) | 14/14 — **100%** |

The fine-tune produces the citation format natively, so a small, fully-GPU-resident
model can replace the spilling 8B for this task with far less post-processing.

## Gotcha: getting the trained model into Ollama

Unsloth's `save_pretrained_gguf` **merged the LoRA to 16-bit but its GGUF
conversion silently failed** on Colab (llama.cpp build) — the zip held only
`model.safetensors`, no `.gguf`. Two ways to finish it locally:

1. **Don't** `ollama create` straight from the safetensors dir — Ollama's own
   converter mangled the tokenizer and the model emitted `???`.
2. **Do** convert with llama.cpp's canonical script, then import the GGUF:
   ```bash
   git clone --depth 1 https://github.com/ggml-org/llama.cpp
   uv pip install gguf sentencepiece protobuf
   CUDA_VISIBLE_DEVICES="" uv run python llama.cpp/convert_hf_to_gguf.py \
       hansard-qwen-gguf --outfile hansard-qwen-f16.gguf --outtype f16
   ollama create hansard-qwen -q q4_K_M -f Modelfile   # FROM ./hansard-qwen-f16.gguf
   ```

**Give the imported model a chat template + stop token.** Creating from a raw
GGUF left `TEMPLATE {{ .Prompt }}` with no stop token, so generation never
terminated (it ran to `num_predict` and got cut mid-answer). Fix: reuse the base
model's template + stops when importing —

```bash
ollama show qwen2.5:1.5b --modelfile > mf   # grab TEMPLATE + PARAMETER stop
sed -i 's|^FROM .*|FROM ./hansard-qwen-f16.gguf|' mf
ollama create hansard-qwen -q q4_K_M -f mf
```

After this the model stops on its own (`done_reason: stop`) with complete answers.
A `num_predict` cap is still kept as a safety net.

## Files
| File | Role |
| --- | --- |
| `build_finetune_data.py` | self-distill the dataset from the production pipeline |
| `train_qlora.py` | Colab QLoRA training + GGUF export |
| `evaluate.py` | base vs fine-tuned format-adherence |
| `train.jsonl` / `val.jsonl` | generated dataset (small; committed as evidence) |
| `Modelfile` | the actual working Modelfile, exported from the deployed model — pairs with the `.gguf` in the `model-v1` GitHub Release (`scripts/fetch_model.sh`) |
