#cbj model
import torch
from peft import LoraConfig, PeftModel, get_peft_model

assert torch.cuda.is_available(), "Cần bật GPU trên Colab."

if isinstance(model, PeftModel):
    raise RuntimeError(
        "Model đã được gắn LoRA. Không chạy lại ô này trên cùng model."
    )

set_seed(42)

# Model gốc dùng FP16 trên T4
model = model.to(device="cuda", dtype=torch.float16)
model.config.use_cache = False

lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    init_lora_weights="gaussian",
    use_rslora=False
)

model = get_peft_model(model, lora_config)

# Trọng số được học giữ ở FP32 để dùng mixed precision FP16
for parameter in model.parameters():
    if parameter.requires_grad:
        parameter.data = parameter.data.float()

# Xác nhận chỉ có trọng số LoRA được học
assert all(
    "lora_" in name
    for name, parameter in model.named_parameters()
    if parameter.requires_grad
)

model.print_trainable_parameters()