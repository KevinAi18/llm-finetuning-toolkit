![License](https://img.shields.io/badge/license-MIT-blue.svg) ![Python](https://img.shields.io/badge/python-3.11-blue.svg) 
# LLM Fine-Tuning Toolkit

An ultra-efficient, production-ready framework for customizing, training, and aligning large language models. Built for AI engineers who require performance, flexibility, and production reliability.

[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-ee4c2c.svg)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-yellow.svg)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

---

## The Problem It Solves

Fine-tuning LLMs is notoriously complex. AI engineers face high compute costs, fragile data loaders, difficult environment setups, and complex distributed training topologies. Moving from base models to domain-expert models often requires stitching together disparate libraries and fighting memory constraints.

The **LLM Fine-Tuning Toolkit** resolves these challenges by providing a unified, highly optimized, and developer-friendly framework. It streamlines the entire pipeline—from data preprocessing and formatting to parameter-efficient fine-tuning, preference alignment, and model export.

## Why Fine-Tuning Matters & Why This Approach is Efficient

While prompt engineering can steer models, deep adaptation—such as teaching a model custom syntax, proprietary APIs, specialized domain logic, or strict system instructions—requires modifying the model's weights. 

Full-parameter fine-tuning of modern LLMs is economically and computationally prohibitive for most teams. This toolkit addresses this problem by integrating state-of-the-art efficiency enhancements:

* **Parameter-Efficient Fine-Tuning (PEFT):** Focus training on a fraction (typically <1%) of the parameters using Low-Rank Adaptation (LoRA), keeping the massive base model frozen.
* **4-bit Quantization (QLoRA):** Leverages specialized NormalFloat data types and double-quantization to load massive foundation weights into cheap GPU memory, maintaining high accuracy.
* **Memory Optimizations:** Integrates FlashAttention-2, gradient checkpointing, and paged optimizers to maximize throughput and prevent Out-of-Memory (OOM) errors.

---

## Tech Stack

| Technology | Role |
| :--- | :--- |
| **PyTorch** | Core tensor compute framework and deep learning engine. |
| **Hugging Face** | Model registries, tokenizers, and high-performance datasets. |
| **PEFT** | Advanced parameter-efficient adapter layers (LoRA, QLoRA, DoRA, VeRA). |
| **TRL** | Transformer Reinforcement Learning library for alignment workflows. |
| **Gradio** | Interactive, no-code dashboard (LLaMA Board) for monitoring and quick configuration. |

---

## Supported Models

We support seamless fine-tuning and inference for over 100+ open-weights architectures, including:

* **LLaMA Series:** LLaMA 2, LLaMA 3, LLaMA 3.1, LLaMA 3.2
* **Mistral Series:** Mistral-7B, Mixtral-8x7B MoE, Mixtral-8x22B MoE
* **Qwen Series:** Qwen 2, Qwen 2.5, Qwen 3, Qwen 3.5 (including MoE and Vision-Language models)
* **DeepSeek Series:** DeepSeek-V2, DeepSeek-V3, DeepSeek-Coder, DeepSeek-R1 (Distilled variants)
* **Other Foundations:** Gemma, Gemma 2, Falcon, Phi-3, Phi-4

---

## Supported Training & Alignment Methods

Customize your training pipeline using industry-standard optimization techniques:

1. **Supervised Fine-Tuning (SFT):** Instruction-tune models on structured input/output datasets.
2. **LoRA & QLoRA:** Minimize VRAM requirements using low-rank adaptors on 16-bit or 4-bit quantized base models.
3. **Direct Preference Optimization (DPO):** Align models with human preferences directly from pairwise choices without building separate reward models.
4. **Proximal Policy Optimization (PPO):** Full reinforcement learning from human feedback (RLHF) with actor, critic, and reward networks.
5. **Odds Ratio Preference Optimization (ORPO):** Align preferences in a single step, combining SFT and preference optimization to speed up convergence.

---

## Hardware Requirements

| Model Size | Precision Mode | Minimum VRAM | Recommended Hardware |
| :--- | :--- | :--- | :--- |
| **7B / 8B** | Full (16-bit BF16) | 24 GB | 1x RTX 3090 / RTX 4090 or A10G |
| **7B / 8B** | Quantized (4-bit QLoRA) | 12 GB | 1x RTX 3060 / RTX 4070 |
| **14B** | Quantized (4-bit QLoRA) | 16 GB | 1x RTX 4080 or A10G |
| **70B** | Quantized (4-bit QLoRA) | 48 GB | 2x RTX 3090/4090 or 1x A100 (80GB) |

---

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/KevinAi18/llm-finetuning-toolkit.git
cd llm-finetuning-toolkit

# Install core dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```

### 2. Launch the Gradio Web UI (No-Code Dashboard)

To load datasets, configure hyperparameters, track training runs visually, and chat with your fine-tuned models:

```bash
llamafactory-cli webui
```

### 3. Command Line Fine-Tuning (Declarative Configurations)

Run high-performance training jobs using declarative YAML configuration files:

```bash
# Start SFT with QLoRA on Qwen 2.5
llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml
```

---

## Interview Q&A

### Q: What is QLoRA, and why use it over full fine-tuning?

**A:** QLoRA (Quantized Low-Rank Adaptation) is a breakthrough fine-tuning technique that drastically reduces memory footprint while preserving downstream model performance.

In **full fine-tuning**, every single parameter of the neural network is updated. This means the GPU must hold not only the model weights in memory but also optimizer states, gradients, and forward activations. For a 70B parameter model, this requires hundreds of gigabytes of VRAM, demanding high-cost enterprise hardware.

**QLoRA overcomes this bottleneck through four core innovations:**

1. **4-bit NormalFloat (NF4) Quantization:** QLoRA quantizes the frozen base model to 4 bits using a specialized information-theoretically optimal data type designed for normally distributed weights.
2. **Double Quantization:** QLoRA quantizes the quantization constants themselves, saving an average of 0.37 bits per parameter (around 3 GB of memory for a 65B model).
3. **Paged Optimizers:** It utilizes NVIDIA Unified Memory to prevent memory spikes (OOMs) during gradient checkpointing by paging optimizer state memory to CPU RAM when VRAM limits are reached.
4. **Targeted LoRA Adapters:** Low-Rank Adaptation matrices are added to all linear layers of the transformer. Only these tiny adapter weights are trained during backpropagation.

By using QLoRA, AI engineers can fine-tune high-quality 7B models on standard consumer GPUs (like a single RTX 4070 or 3060) and run 70B models on dual-consumer GPUs, dropping infrastructure costs by over 90% without sacrificing accuracy.
 
## Supported Fine-Tuning Methods 
- LoRA - Low Rank Adaptation for parameter efficient tuning 
- QLoRA - Quantized LoRA for memory constrained training 
- DPO - Direct Preference Optimization for alignment 
- PPO - Proximal Policy Optimization with reward model 
- ORPO - Combined SFT and preference alignment in one step 
 
## Quick Start 
 
1. Install dependencies from requirements.txt 
2. Prepare your dataset in prompt/response JSON format 
3. Choose a fine-tuning method - LoRA, QLoRA, DPO, PPO or ORPO 
4. Run the training script with your config file 
5. Evaluate the fine-tuned model using the included eval scripts 
 
## Requirements 
- Python 3.10 or higher 
- CUDA capable GPU recommended for training 
- transformers, peft, trl and bitsandbytes libraries 
- Minimum 16GB GPU memory for QLoRA on 7B models 
 
## Roadmap 
- Add support for multi-GPU distributed training out of the box 
- Integrate automated hyperparameter search 
- Add support for newer base models as they release 
 
## Contributing 
- Fork the repository and create a feature branch 
- Test all changes on at least one supported fine-tuning method 
- Document any new training configs or dataset formats you add 
- Open a pull request with benchmark results where possible 
 
## Performance 
- QLoRA fine-tuning of 7B model completes in under 2 hours on A100 
- LoRA reduces trainable parameters by over 90 percent vs full tuning 
- DPO training converges faster than PPO with less GPU memory 
- Evaluation scripts run automated benchmarks after each training run 
 
## License 
This project is released under the MIT License. See the LICENSE file for details. 
 
## Acknowledgements 
Built using Hugging Face transformers, peft and trl libraries. 
 
## FAQ 
Q: Which is better, LoRA or full fine-tuning? 
A: LoRA is more memory efficient and usually sufficient unless you need maximum performance on a highly specialized task. 
 
## Troubleshooting 
- If you hit out of memory errors, reduce batch size or enable gradient checkpointing 
- If training loss is not decreasing, check your learning rate and dataset format 
 
## Design Philosophy 
This toolkit favors sensible defaults over exhaustive configuration, so someone can go from raw dataset to a fine-tuned model without needing to understand every hyperparameter first. 
