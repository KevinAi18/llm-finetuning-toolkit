 
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
