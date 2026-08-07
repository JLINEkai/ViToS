# 🩺 Token-Sparse Medical Multimodal Reasoning via Dual-Stream Reinforcement Learning

<div align="center">

### 🧠 ViToS: Visual Token-Sparse Reasoning for Medical Multimodal Models

[![ICML 2026](https://img.shields.io/badge/ICML-2026-0D5C9B?logo=icml&logoColor=white)](https://icml.cc/)
[![Paper](https://img.shields.io/badge/arXiv-2606.31599-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2606.31599)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ViTAR--16K-FFD21E)](https://huggingface.co/datasets/jline/ViTAR-16K)

**ViToS** (**Vi**sual **To**ken-**S**parse reasoning) is a dual-stream reinforcement learning framework for efficient and evidence-grounded medical multimodal reasoning.



</div>

> 🚧 **Release status:** The codebase is actively being organized. Additional training, inference, documentation, and reproducibility resources will be released over the coming weeks.

## 🔍 Overview

Medical vision-language models commonly process all visual tokens throughout reasoning, even when only a small image region is relevant to the clinical question. This introduces redundant computation and can distract the model from diagnostically important evidence.

ViToS reformulates visual token pruning (VTP) as a **policy-guided evidence selection process** and optimizes evidence localization and token-sparse reasoning end to end. A single shared policy is trained through two cascaded branches:

1. **Localization branch** — predicts spatial grounding for the evidence relevant to the question.
2. **Token-sparse reasoning branch** — converts the predicted grounding into a compact visual-token representation and performs complete medical reasoning using the selected evidence.

The two branches are coupled: localization determines which evidence is available to sparse reasoning, while downstream reasoning quality provides a learning signal for evidence selection. ViToS addresses this dependency with **cross-feedback sequential optimization**, reducing gradient conflict and improving convergence of the shared policy model.

## 🧠 Method overview

<p align="center">
  <img src="assets/framework.png" alt="ViToS dual-stream reinforcement learning framework" width="100%">
</p>

### ✨ Key features

- **Policy-guided VTP:** visual evidence selection is learned as part of the reasoning policy instead of being a fixed preprocessing heuristic.
- **Dual-stream learning:** one shared policy supports spatial localization and downstream token-sparse reasoning.
- **Grounding-aware sparsification:** predicted bounding boxes are mapped to the visual-token grid, retaining question-relevant regions for the next reasoning pass.
- **Cross-feedback sequential optimization:** the two coupled objectives are optimized in sequence to stabilize shared-policy learning.
- **End-to-end RL training:** accuracy, output format, grounding quality, and cascaded reasoning signals are integrated into the training workflow.
- **Distributed training stack:** Ray, FSDP, vLLM, and GRPO are used for scalable multimodal RL training.

## 🗂️ Repository layout

```text
ViToS/
├── data/                              # Parquet shards, JSON annotations, images, embeddings
├── examples/
│   ├── grounding.yaml                # Main ViToS training configuration
│   ├── grounding_train_stage1.sh     # Localization-stage launcher
│   ├── grounding_train_stage2.sh     # Token-sparse reasoning launcher
│   ├── format_prompt/                 # Prompt templates
│   └── reward_function/              # Accuracy, format, and grounding rewards
├── scripts/
│   └── convert_hf_parquet_to_train_json.py
├── verl/
│   ├── trainer/                       # Dual-stage RL dataflow and optimization
│   ├── workers/                       # FSDP and rollout workers
│   └── utils/                         # Dataset and multimodal utilities
├── process_visual_token_pretrain.py  # Full-image visual-token precomputation
└── requirements.txt
```

## ⚙️ Installation

### 🧩 Requirements

- Python 3.11 or newer
- CUDA-capable NVIDIA GPUs
- A CUDA-compatible PyTorch installation


### 🚀 Set up the environment

```bash
git clone https://github.com/JLINEkai/ViToS.git
cd ViToS

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install PyTorch for the CUDA version available on your machine, then install ViToS and its remaining dependencies:

```bash
pip install -e .
```

> `flash-attn`, vLLM, and PyTorch must be mutually compatible with the installed CUDA toolkit. Installing PyTorch first generally makes build failures easier to diagnose.

## 🗃️ Data preparation

### 🤗 ViTAR-16K dataset

The ViTAR-16K dataset is available on Hugging Face: [**jline/ViTAR-16K**](https://huggingface.co/datasets/jline/ViTAR-16K). Please follow the dataset card for access, usage, and license information. After downloading the Parquet shards into `data/`, use the conversion and preprocessing steps below to prepare the local training layout.

### 🧾 Expected annotation format

ViToS uses a JSON list with one image, question, answer, and one or more normalized bounding boxes per sample:

```json
[
  {
    "image_path": [
      "data/images/2754.jpg"
    ],
    "question": "<image>What is the type of tumor at image coordinates (0.4, 0.55)? A) Glioma B) Meningioma C) Pituitary D) Other normal area",
    "answer": "C",
    "bbox": [
      [
        0.33,
        0.50,
        0.45,
        0.62
      ]
    ]
  }
]
```

The default launchers read:

```text
data/train_data.json
```

### 🔄 Convert Hugging Face Parquet shards

If the dataset is stored as Hugging Face Parquet shards with `image`, `question`, `answer`, and `bbox` columns, run:

```bash
python scripts/convert_hf_parquet_to_train_json.py
```

The converter will:

- read `data/train-*.parquet` in shard order;
- extract embedded image bytes into `data/images/`;
- deduplicate identical images and resolve filename collisions;
- add the `<image>` marker when it is absent; and
- write `data/train_data.json` in the format shown above.

Custom input and output locations can be supplied through CLI arguments:

```bash
python scripts/convert_hf_parquet_to_train_json.py \
  --input-dir /path/to/parquet_shards \
  --pattern 'train-*.parquet' \
  --output-images data/images \
  --output-json data/train_data.json \
  --json-image-prefix data/images
```

## 🧮 Precompute visual tokens

ViToS reuses full-image visual embeddings while constructing grounding-aware sparse token sequences. Precompute embeddings with the same model and image-resolution settings that will be used during training:

```bash
python process_visual_token_pretrain.py \
  --data-dir data \
  --model-path /path/to/Lingshu-7B \
  --output-dir data/embeddings \
  --max-samples 0
```

`--max-samples 0` processes the complete dataset. The script default is intentionally limited to 100 rows for a quick preprocessing check.

The current preprocessing utility reads local Parquet shards and expects the embedded image column to be named `image`. It writes one rank-2 tensor per unique image:

```text
data/embeddings/<image-stem>_visual.pt
```

Training verifies that every embedding has the visual-token count expected from the image grid. Use the same values for `--model-path`, `--min-pixels`, and `--max-pixels` during preprocessing and training.

The resulting data directory should look like:

```text
data/
├── train-00000-of-00003.parquet
├── train-00001-of-00003.parquet
├── train-00002-of-00003.parquet
├── train_data.json
├── images/
│   ├── 1.jpg
│   └── ...
└── embeddings/
    ├── 1_visual.pt
    ├── ...
    └── manifest.json
```

## 🎯 Training

ViToS is trained sequentially. Stage 2 resumes from the Stage-1 checkpoint and continues optimization with the grounding-aware token-sparse branch.

### 📍 Stage 1: evidence localization

```bash
MODEL_PATH=/path/to/Lingshu-7B \
STAGE1_SAVE_DIR=/path/to/checkpoints/vitos_stage1 \
bash examples/grounding_train_stage1.sh
```

Stage 1 trains the shared policy to produce structured medical reasoning, normalized evidence boxes, and the final answer. Its reward combines answer accuracy, response-format compliance, grounding overlap, and the cascaded auxiliary reasoning signal.

### 🧠 Stage 2: token-sparse reasoning

After Stage 1 finishes, launch Stage 2 with the same Stage-1 save directory:

```bash
MODEL_PATH=/path/to/Lingshu-7B \
STAGE1_SAVE_DIR=/path/to/checkpoints/vitos_stage1 \
STAGE2_SAVE_DIR=/path/to/checkpoints/vitos_stage2 \
bash examples/grounding_train_stage2.sh
```

By default, the Stage-2 launcher reads `checkpoint_tracker.json` from `STAGE1_SAVE_DIR` and resumes the most recent actor checkpoint. A checkpoint can also be selected explicitly:

```bash
MODEL_PATH=/path/to/Lingshu-7B \
STAGE1_CHECKPOINT_PATH=/path/to/stage1/checkpoint \
STAGE2_SAVE_DIR=/path/to/checkpoints/vitos_stage2 \
bash examples/grounding_train_stage2.sh
```

Stage 2 uses the first-round grounding to select and merge visual tokens, then optimizes complete reasoning and answer generation over the compact visual representation.



## 🩹 Troubleshooting

### 🔎 `Visual embedding not found`

Run preprocessing with `--max-samples 0`. Image stems in `train_data.json` must match the corresponding `<image-stem>_visual.pt` filenames.

### ⚠️ Visual embedding/image token-count mismatch

Preprocessing and training must use the same model processor, `min_pixels`, `max_pixels`, and image files. Regenerate the embeddings after changing any of them.

### 💾 Stage-2 checkpoint tracker not found

Set `STAGE1_SAVE_DIR` to the Stage-1 output directory, or pass `STAGE1_CHECKPOINT_PATH` explicitly.

### 🖥️ CUDA, FlashAttention, or vLLM installation errors

Check that PyTorch, CUDA, FlashAttention, and vLLM were built for compatible versions. These packages are sensitive to CUDA and compiler mismatches.

## 🧑‍💻 Development

```

The launch scripts can be checked without starting training:

```bash
bash -n examples/grounding_train_stage1.sh
bash -n examples/grounding_train_stage2.sh
```

## 📚 Citation

If you find ViToS useful, please cite:

```bibtex
@article{chen2026token,
  title={Token-Sparse Medical Multimodal Reasoning via Dual-Stream Reinforcement Learning},
  author={Chen, Kaitao and Zhao, Weiqian and Wu, Jiamin and Zheng, Qihao and Sun, Shangquan and Song, Chunfeng and Wang, Xiaosong and Zhou, Mu and Liu, Mianxin},
  journal={arXiv preprint arXiv:2606.31599},
  year={2026}
}
```

## 🙏 Acknowledgements

ViToS uses a veRL-style distributed reinforcement learning stack and builds on the open-source PyTorch, Hugging Face Transformers, Ray, FSDP, and vLLM ecosystems. Source files derived from upstream projects retain their original copyright and license notices.
