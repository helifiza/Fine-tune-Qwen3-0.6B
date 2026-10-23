def evaluate_each_sample(disable_lora=False):
    losses = []

    context = (
        eval_model.disable_adapter()
        if disable_lora
        else nullcontext()
    )

    name = "Qwen gốc" if disable_lora else "Qwen sau fine-tune"

    with context, torch.inference_mode():
        for token_ids, prompt_length in tqdm(encoded_val, desc=name):
            input_ids = torch.tensor(
                [token_ids],
                dtype=torch.long,
                device=device,
            )

            labels = input_ids.clone()
            labels[:, :prompt_length] = -100

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = eval_model(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    labels=labels,
                    use_cache=False,
                ).loss.float().item()

            if not math.isfinite(loss):
                raise RuntimeError(
                    f"Loss không hợp lệ ở mẫu {len(losses) + 1}: {name}"
                )

            losses.append(loss)

    return losses


# Tắt adapter: đánh giá Qwen gốc.
loss_before = evaluate_each_sample(disable_lora=True)

# Adapter tự được bật lại sau khi thoát context ở trên.
loss_after = evaluate_each_sample(disable_lora=False)

results = pd.DataFrame({
    "sample_id": range(1, len(encoded_val) + 1),
    "loss_before": loss_before,
    "loss_after": loss_after,
    "answer_tokens": [
        len(token_ids) - prompt_length
        for token_ids, prompt_length in encoded_val
    ],
})

results["loss_reduction"] = (
    results["loss_before"] - results["loss_after"]
)

# Loss toàn tập được tính theo tổng số token đáp án.
summary = []

for name, column in [
    ("Qwen gốc", "loss_before"),
    ("Qwen sau fine-tune", "loss_after"),
]:
    mean_loss = (
        (results[column] * results["answer_tokens"]).sum()
        / results["answer_tokens"].sum()
    )

    summary.append({
        "Model": name,
        "Số mẫu": len(results),
        "Loss": mean_loss,
        "Perplexity": math.exp(mean_loss),
    })

metrics = pd.DataFrame(summary)
display(metrics.round(4))

results.to_csv(
    "/content/so_sanh_tung_cau.csv",
    index=False,
    encoding="utf-8-sig",
)

metrics.to_csv(
    "/content/so_sanh_metrics.csv",
    index=False,
    encoding="utf-8-sig",
)