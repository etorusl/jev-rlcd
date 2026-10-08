# RLCD Stage 1 — Calibrated Readout for Agent-Output Auditing

Training a language model to **audit an agent's final response** for hallucinations,
fake tool calls and policy violations in long interaction/tool logs, using
**RLCD Stage 1**: the model samples its own chain-of-thought rationales
(Monte-Carlo), and a **Brier-calibrated readout** is trained on those rationales.
No policy-gradient / REINFORCE signal is applied to the reasoning tokens — the
gradient only flows from the two-way answer readout.

Built with 🤗 Transformers + PEFT (LoRA) + bitsandbytes + Accelerate, targeting
**1× NVIDIA H100 80GB** with up to **~20k-token** (hard cap 50k) contexts.

---

## Method (per sample)

1. **Base prompt** — task template (`system` + `user`) ending with the assistant
   header, containing `{prompt}` (interaction history / tool logs) and
   `{response}` (the agent answer to audit).
2. **Monte-Carlo reasoning** — `model.generate()` under `torch.no_grad()` with
   `temperature`, `max_new_tokens`, stopping on the `Answer:` marker. Produces
   `M` independent rationales `CoT_i`, each ending at the marker.
3. **Full sequence** — `Full_Input_i = base + CoT_i` (the marker is already the
   last token; no extra tags). Built at the token level for exactness.
4. **Readout forward** — a gradient-enabled forward pass per `Full_Input_i`;
   take the logits at the **last position** (immediately after `Answer:`).
5. **Two-way readout** — keep the tokens for class `1` and class `0`, softmax →
   `u_i = [u_{i,1}, u_{i,0}]` (surface variants like `"1"` and `" 1"` are merged
   in log-space so a leading space never breaks the readout).
6. **Brier loss** — `J(u_i, Y) = 2 u_{i,Y} − ||u_i||²`, batch loss
   `L = −(1/M) Σ_i J(u_i, Y)`. `loss.backward()` updates LoRA + the head.

Micro-batch is fixed at **1**; **gradient accumulation** reaches the real batch
(default 8 samples → 8 accumulation steps). Gradient checkpointing is enabled and
LoRA covers **all Linear layers**, with `lm_head` fully trainable
(`modules_to_save=["lm_head"]`) — the readout head is exactly what calibration
needs to move.

---

## Repository layout

```
train.py                     # entry point
rlcd/
  config.py                  # dataclass config + YAML/override loading
  prompts.py                 # PromptBuilder: base text, marker, target token ids
  data.py                    # loading, oversize filtering, splits, batching
  modeling.py                # model/tokenizer, quantization, LoRA, grad-ckpt
  sampling.py                # MC generation + marker stopping criteria
  losses.py                  # class log-scores, Brier reward/loss
  metrics.py                 # acc / Brier / AUROC
  trainer.py                 # training + eval + checkpoint loop
  utils.py                   # param counting, memory logging
configs/
  audit_qwen.yaml            # main task (Qwen, 8-bit, long context)
  smoke.yaml                 # fast end-to-end sanity run on the real model (from cache)
scripts/
  setup_cluster.sh           # venv + deps + CUDA check
  run_train.sh               # launch (single or multi-GPU via accelerate)
```

---

## Data

`samples.json` is a JSON array of objects:

```json
{
  "id": "...", "group": "...", "domain": "telecom", "turn": 19,
  "prompt": "⟦SYSTEM⟧ ... full dialogue + tool calls ...",
  "response": "final agent answer to audit",
  "label": 1
}
```

* `label` `1` = hallucination / fake tool / policy violation, `0` = clean.
* The file is ~249 MB and therefore **gitignored** (GitHub's 100 MB limit).
  Copy it to the cluster (e.g. `scp`) or point `--data_path` at it.

### No truncation policy

`data.max_seq_len: 50000` is a **hard cap**, and `oversize_policy: drop` means
samples whose base input exceeds it are **dropped**, never cut — we do not want
truncated samples in the training set. The filter pass is cached under the output
directory. (For other tasks you can set `oversize_policy: truncate` with
`truncate_side: head|tail|middle`.)

---

## Quick start (cluster, H100)

```bash
git pull
bash scripts/setup_cluster.sh
source .venv/bin/activate

# ensure samples.json is present in the repo root (it is gitignored)
python train.py --config configs/audit_qwen.yaml
```

Common overrides (no editing needed):

```bash
# exact model id
python train.py --config configs/audit_qwen.yaml --model Qwen/Qwen3-8B
# full-precision training instead of 8-bit
python train.py --config configs/audit_qwen.yaml --set model.quantization=none
# different Monte-Carlo count / batch
python train.py --config configs/audit_qwen.yaml --set train.M=2 --set train.real_batch_size=4
```

Sanity-check the whole pipeline on the real model first (2 steps + eval + save,
8-bit from cache, tiny decoding budget). Run it on the GPU node:

```bash
python train.py --config configs/smoke.yaml
```

---

## Key configuration knobs

| Area | Key | Notes |
| --- | --- | --- |
| Model | `model.model_name_or_path` | `Qwen/Qwen3.5-9B` by default; `trust_remote_code` if needed |
| Precision | `model.quantization` | `8bit` (default) / `4bit` / `none` (bf16 full precision) |
| Attention | `model.attn_implementation` | `sdpa` (dispatches to the flash kernel when available) |
| LoRA | `model.lora_target_modules` | `null` → every Linear layer |
| Readout head | `model.modules_to_save` | `["lm_head"]` (fully trained) — or set `lora_includes_lm_head: true` |
| Readout scope | `model.train_scope` | `lora_plus_head` (default) / `lora_only` / `head_only` |
| MC sampling | `train.M`, `train.max_new_tokens`, `train.temperature` | `M` rationales per sample |
| Batch | `train.micro_batch_size=1`, `train.real_batch_size` | accumulation is derived |
| Cost | `train.eval_every`, `train.save_every`, `train.max_steps` | long-context generation is the bottleneck |
| Thinking | `prompt.disable_thinking` + `disable_thinking_phrase` | appends ` /no_think` for hybrid-reasoning models |

---

## Efficiency & robustness notes

* **Memory.** 8-bit base + LoRA + gradient checkpointing keeps a ~9B model,
  a 20k-token context and `M` readout passes within 80 GB. Sampling is
  **sequential by default** (`sampling_mode: sequential`) so only one
  sequence's KV cache is live at a time; set `sampling_mode: batched` to share
  the prefill across the `M` samples when memory allows.
* **OOM.** `train.skip_oom: true` catches CUDA OOM per micro-step, frees the
  cache and skips that accumulation group instead of crashing.
* **Answer token resolution.** `PromptBuilder` auto-resolves the token ids of
  class `1`/`0` as continuations of the marker and logs them at startup, e.g.
  `Answer token(s) for class 1: [(16, '1'), (...)]`. Surface variants `"1"` / `" 1"`
  are merged via `logsumexp` so a leading space cannot break the readout. You can
  hard-override ids with `label.label_token_ids`.
* **Universality.** The template, marker, class labels/strings and every
  decoding/training parameter live in the config — swap `configs/*.yaml` for a
  new task without touching code.

---

## Outputs & evaluation

Checkpoints (adapter / head + tokenizer + `rlcd_config.json`) are written to
`train.output_dir/step_*` and `.../final`. Per-sample probabilities are dumped to
`<output_dir>/eval/step_<N>.jsonl` (with a `_metrics.json` alongside) so you can
inspect mistakes offline.

Evaluation (`eval_every` steps and at the end) reports:

| Metric | Meaning | Desired |
| --- | --- | --- |
| `brier` | Brier score = mean squared error of the predicted class probabilities vs the 0/1 label (the core RLCD objective) | ↓ |
| `brier_baseline` | Brier of a constant predictor at the class base rate | reference |
| `aurc` | Area under the risk–coverage curve: how well confidence correlates with actual errors | ↓ |
| `cov@0.05`, `cov@0.10` | Coverage achievable while keeping selective error ≤ 5% / 10% | ↑ |
| `ece` | Expected calibration error | ↓ |
| `auroc` | ROC-AUC of P(hallucination) | ↑ |
| `pr_auc` | PR-AUC (average precision) — more informative under class imbalance (~9% positives) | ↑ |
| `f1`, `precision`, `recall` | Positive-class (hallucination) detection | ↑ |
| `acc`, `acc_majority` | Accuracy vs. majority-class baseline | ↑ |

If `brier` does not fall below `brier_baseline`, or `auroc`/`pr_auc` stay at
chance, the readout has not learned to associate the model's own rationales with
the answer. Watch `aurc`/`cov@*` specifically: they are what RLCD Stage 1 targets.

## Troubleshooting

**`AttributeError: module 'torch' has no attribute 'accelerator'` / `Could not
import module 'GenerationMixin'`.** A torch/transformers version mismatch: recent
`transformers` import `torch.accelerator`, added in **torch 2.6**. Reinstall with

```bash
pip install "torch>=2.6.0" --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python scripts/check_env.py
```

If the driver cannot run CUDA 12.4 wheels, pick a matching index
(`cu126`, `cu121`, …) via `TORCH_INDEX_URL=... bash scripts/setup_cluster.sh`.
`scripts/check_env.py` prints the whole stack and flags this exact problem up front.

**Offline / cached models.** If the node has no HF network access, set
`model.local_files_only: true` (as in `configs/smoke.yaml`) so weights load from
the local HF cache; you can also export `HF_HUB_OFFLINE=1`.

**Disk space.** The HF cache defaults to `~/.cache/huggingface`. If that volume is
full you will see `Not enough free disk space to download the file`. Point it at a
roomy disk before running (Qwen3.5-9B is ~20 GB in bf16):

```bash
export HF_HOME=/workspace/data/hf_cache   # put on a volume with space
```

## Qwen3.5 specifics

`Qwen/Qwen3.5-9B` is a very new checkpoint and differs from Qwen3 in ways that
matter here:

* **Architecture is unknown to pip `transformers`** (`model_type: qwen3_5`) →
  install transformers from `main` (see Troubleshooting / setup script).
* **It is multimodal + hybrid** (vision encoder + Gated DeltaNet/attention +
  sparse MoE). It loads via `AutoModelForMultimodalLM` rather than
  `AutoModelForCausalLM`. This is exposed as `model.auto_model_class`:
  ```bash
  python train.py --config configs/audit_qwen.yaml --set model.auto_model_class=AutoModelForMultimodalLM
  ```
  If a text-only forward pass fails on the multimodal wrapper, use a text-only
  base model instead or a vLLM `--language-model-only` style export.
* **There is no soft `/think` `/nothink` switch** (unlike Qwen3). Thinking is
  toggled through the chat template. Our default `Raw` template cannot disable
  it, so for Qwen3.5 prefer:
  ```yaml
  prompt:
    mode: chat_template
    disable_thinking: true   # -> enable_thinking=False in apply_chat_template
  ```
* **LoRA over "all linear layers" can be huge** because MoE expert projections are
  Linear too. If the adapter explodes in size or OOMs, exclude them:
  ```bash
  --set model.lora_exclude_modules='["experts","router"]' \
  --set model.lora_target_modules='["q_proj","k_proj","v_proj","o_proj"]'
  ```
  (Inspect the real names first with `--set` off + the startup log, or a 5-line
  `named_modules()` print.)
* **8-bit bitsandbytes** may not yet support this hybrid architecture; if
  quantization fails, fall back to full precision: `--set model.quantization=none`
  (bf16 needs ~20 GB of weights on an 80 GB H100).



