import json, math, os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer

OUTPUT_DIR = "/content/qwen3_luat_checkpoints"
LOG_FILE = f"{OUTPUT_DIR}/train_log.jsonl"
EVAL_STEPS = 25
os.makedirs(OUTPUT_DIR, exist_ok=True)


class StepTracker(TrainerCallback):
    """In và ghi lại chi tiết từng step ra màn hình + file JSONL."""

    def __init__(self, log_file):
        self.log_file = log_file
        self.records = []

    def on_train_begin(self, args, state, control, **kwargs):
        self.start = self.last = time.time()
        torch.cuda.reset_peak_memory_stats()
        open(self.log_file, "w").close()
        samples_per_step = args.per_device_train_batch_size * args.gradient_accumulation_steps
        print(f"Bắt đầu train: {state.max_steps} steps, mỗi step = {samples_per_step} mẫu")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        now = time.time()
        record = {
            "step": state.global_step,
            "epoch": round(state.epoch or 0.0, 3),
            "elapsed_s": round(now - self.start, 1),
            "step_time_s": round(now - self.last, 2),
            "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
            **logs,
        }
        self.last = now
        self.records.append(record)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        nan = float("nan")
        if "eval_loss" in logs:
            # Bản ghi eval: không tính thời gian eval vào step kế tiếp
            self.last = time.time()
            eval_loss = logs["eval_loss"]
            print(
                f"   ↳ EVAL @ step {state.global_step}: "
                f"eval_loss={eval_loss:.4f} | ppl={math.exp(min(eval_loss, 20)):.2f} | "
                f"acc={logs.get('eval_mean_token_accuracy', nan):.3f} | "
                f"{logs.get('eval_runtime', 0):.0f}s"
            )
        elif "loss" in logs:
            loss = logs["loss"]
            grad_norm = logs.get("grad_norm") or 0.0
            eta_min = (state.max_steps - state.global_step) * record["elapsed_s"] / max(state.global_step, 1) / 60
            print(
                f"[step {state.global_step:>4}/{state.max_steps}] "
                f"loss={loss:.4f} | grad_norm={grad_norm:.3f} | "
                f"lr={logs.get('learning_rate', 0):.2e} | "
                f"acc={logs.get('mean_token_accuracy', nan):.3f} | "
                f"{record['step_time_s']:.1f}s/step | "
                f"GPU={record['gpu_peak_gb']}GB | ETA≈{eta_min:.1f} phút"
            )
            if not math.isfinite(loss):
                print("⚠️ Loss là NaN/Inf, dừng train.")
                control.should_training_stop = True
            elif not math.isfinite(grad_norm):
                # Với fp16, grad_norm=inf ở vài step đầu là bình thường (GradScaler đang hiệu chỉnh)
                print("   ⚠️ grad_norm = inf/NaN (step này có thể bị bỏ qua bởi fp16 scaler)")

    def on_save(self, args, state, control, **kwargs):
        print(f"   💾 Đã lưu checkpoint tại step {state.global_step}")


PER_DEVICE_BS = 8      # trước: 1
GRAD_ACCUM = 1         # trước: 8 → batch hiệu dụng vẫn là 8, nên cách học không đổi
EVAL_STEPS = 100       # trước: 25
EVAL_SAMPLES = 500     # trước: toàn bộ 2.999 mẫu

# Cắt ở phân vị 99 để vài mẫu quá dài không gây OOM khi gom batch
MAX_LEN_TRAIN = int(np.percentile(lengths, 99))
print("Phân vị 50/90/99/100:", [int(np.percentile(lengths, p)) for p in (50, 90, 99, 100)])
print("max_length dùng để train:", MAX_LEN_TRAIN)

# Eval trên tập con ngẫu nhiên
sft_val_small = sft_val.shuffle(seed=42).select(range(min(EVAL_SAMPLES, len(sft_val))))

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,

    num_train_epochs=1,
    per_device_train_batch_size=PER_DEVICE_BS,
    gradient_accumulation_steps=GRAD_ACCUM,

    learning_rate=1e-4,
    optim="adamw_torch",
    max_grad_norm=1.0,

    max_length=MAX_LEN_TRAIN,
    packing=False,
    completion_only_loss=True,
    eos_token="<|im_end|>",

    fp16=True,
    bf16=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},

    logging_strategy="steps",
    logging_steps=1,
    logging_first_step=True,
    logging_nan_inf_filter=False,

    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    per_device_eval_batch_size=PER_DEVICE_BS,

    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,

    report_to="tensorboard",
    seed=42,
)

steps_per_epoch = math.ceil(len(sft_train) / (PER_DEVICE_BS * GRAD_ACCUM))
print(f"Train: {len(sft_train)} mẫu | Eval: {len(sft_val_small)} mẫu | ≈{steps_per_epoch} steps/epoch")

tracker = StepTracker(LOG_FILE)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=sft_train,
    eval_dataset=sft_val_small,
    processing_class=tokenizer,
    callbacks=[tracker],
)

train_result = trainer.train()
print(train_result.metrics)