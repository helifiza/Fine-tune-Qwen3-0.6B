import math
import torch
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from contextlib import nullcontext
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm
from IPython.display import display

assert torch.cuda.is_available(), "Hãy bật GPU trong Colab."

# Cần đúng tập validation đã chia trước khi train.
assert "sft_val" in globals(), (
    "Chưa có sft_val. Hãy chạy lại đúng các ô chia dữ liệu và "
    "định dạng validation trước đó; không cần chạy lại ô train."
)
assert len(sft_val) > 0
assert {"prompt", "completion"}.issubset(sft_val.column_names)

# Trường hợp 1: Model vẫn còn trong phiên Colab vừa train xong.
if "trainer" in globals() and isinstance(trainer.model, PeftModel):
    eval_model = trainer.model
    print("Sử dụng model đang có trong phiên Colab.")

# Trường hợp 2: Nạp lại model gốc và adapter đã lưu.
else:
    ADAPTER_DIR = Path("/content/qwen3_luat_lora_adapter")

    assert (ADAPTER_DIR / "adapter_config.json").exists(), (
        "Không tìm thấy adapter. Hãy tải ZIP lên Colab và giải nén "
        "để có /content/qwen3_luat_lora_adapter/adapter_config.json"
    )

    tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_DIR))

    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        dtype=torch.float16,
        device_map={"": "cuda:0"},
    )

    eval_model = PeftModel.from_pretrained(
        base_model,
        str(ADAPTER_DIR),
        is_trainable=False,
    )

    print("Đã nạp lại Qwen gốc và adapter đã lưu.")

eval_model.eval()
device = eval_model.get_input_embeddings().weight.device
assert device.type == "cuda"

print("Số mẫu validation:", len(sft_val))
print("Thiết bị:", device)

# Chuẩn bị dữ liệu một lần để cả hai model dùng giống nhau.
encoded_val = []

for sample in tqdm(sft_val, desc="Chuẩn bị toàn bộ validation"):
    prompt_ids = tokenizer(
        sample["prompt"],
        add_special_tokens=False,
    )["input_ids"]

    full_ids = tokenizer(
        sample["prompt"] + sample["completion"],
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]

    # Xác định phần prompt để chỉ tính loss trên đáp án.
    prompt_length = 0
    for a, b in zip(prompt_ids, full_ids):
        if a != b:
            break
        prompt_length += 1

    assert 0 < prompt_length < len(full_ids), "Mẫu không có token đáp án."
    assert len(full_ids) <= eval_model.config.max_position_embeddings

    encoded_val.append((full_ids, prompt_length))