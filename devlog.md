 
## 2026-06-15 
### LoRA Fine-Tuning Notes 
- LoRA adds small trainable matrices to frozen model weights 
- Reduces trainable parameters by 90 percent vs full fine-tuning 
- Rank r controls size of LoRA matrices, lower r = less params 
- Alpha controls scaling of LoRA updates during training 
 
## 2026-06-17 
### QLoRA Fine-Tuning Notes 
- QLoRA combines quantization with LoRA for memory efficiency 
- 4-bit quantization reduces GPU memory by up to 75 percent 
- bitsandbytes library handles 4-bit quantization in Python 
- QLoRA allows fine-tuning 70B models on single consumer GPU 
 
## 2026-06-18 
### DPO Fine-Tuning Notes 
- DPO trains model on preferred vs rejected response pairs 
- No reward model needed unlike PPO making training simpler 
- Dataset format requires prompt, chosen and rejected columns 
- DPO more stable than PPO and uses less GPU memory 
 
## 2026-06-21 
### ORPO Fine-Tuning Notes 
- ORPO combines SFT and preference alignment in single training step 
- No need for separate reward model or reference model 
- Odds ratio penalty discourages rejected response generation 
- ORPO uses less GPU memory and trains faster than DPO 
 
## 2026-06-23 
### PPO Fine-Tuning with RLHF Notes 
- PPO uses reward model to score LLM responses during training 
- KL divergence penalty stops model drifting too far from base 
- Actor critic architecture used to stabilize PPO training 
- TRL library provides ready to use PPO trainer for LLMs 
 
## 2026-06-25 
### Dataset Preparation for Fine-Tuning Notes 
- Instruction tuning datasets need prompt and response pairs 
- Data deduplication removes near identical training examples 
- Quality filtering removes low quality or toxic samples 
- Tokenization length distribution checked before training starts 
 
## 2026-06-28 
### Gradient Checkpointing Notes 
- Gradient checkpointing trades compute for reduced memory usage 
- Only stores subset of activations during forward pass 
- Recomputes missing activations during backward pass when needed 
- Allows fine-tuning larger models on limited GPU memory 
 
## 2026-06-30 
### Evaluation Metrics for Fine-Tuned Models 
- Perplexity measures how well model predicts held out test set 
- BLEU and ROUGE scores used for translation and summarization tasks 
- Human evaluation still gold standard for response quality checks 
- LLM as judge used for scalable automated response evaluation 
