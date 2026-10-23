from transformers import set_seed

set_seed(42)

SYSTEM_PROMPT = (
    "Bạn là trợ lý hỏi đáp pháp luật. Trả lời bằng tiếng Việt."
)

# Cấu hình tokenizer
tokenizer.padding_side = "right"
tokenizer.eos_token = "<|im_end|>"

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# Lấy câu hỏi từ dữ liệu đã tách cột hoặc cột text gốc
def get_question(sample):
    if "question" in sample:
        question = sample["question"].strip()
    else:
        user_text = sample["text"].split(
            "<|im_start|>user\n", 1
        )[1]
        user_text = user_text.split("<|im_end|>", 1)[0]
        question = user_text.split("### Question", 1)[1].strip()

    if not question:
        raise ValueError("Có mẫu thiếu câu hỏi.")

    return question


# Lấy đáp án cuối, bỏ phần <think>...</think>
def get_answer(sample):
    if "answer" in sample:
        answer = sample["answer"].strip()
    else:
        assistant_text = sample["text"].split(
            "<|im_start|>assistant\n", 1
        )[1]
        answer = assistant_text.split("<|im_end|>", 1)[0].strip()

    answer = answer.split("</think>", 1)[-1].strip()

    if not answer:
        raise ValueError("Có mẫu thiếu đáp án.")

    return answer


# Chuyển mỗi mẫu thành prompt và completion
def format_sample(sample):
    question = get_question(sample)
    answer = get_answer(sample)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    completion = answer + tokenizer.eos_token

    # Đếm toàn bộ token: prompt + đáp án + token kết thúc
    token_ids = tokenizer(
        prompt + completion,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]

    return {
        "prompt": prompt,
        "completion": completion,
        "n_tokens": len(token_ids),
    }


def prepare(dataset, description):
    return dataset.map(
        format_sample,
        remove_columns=dataset.column_names,
        desc=description,
    )


assert len(train_raw) > 0, "Tập train đang trống."
assert len(validation_raw) > 0, "Tập validation đang trống."

# Chuẩn bị riêng hai tập đã chia trước đó
prepared_train = prepare(train_raw, "Định dạng tập train")
prepared_val = prepare(validation_raw, "Định dạng tập validation")

train_lengths = prepared_train["n_tokens"]
val_lengths = prepared_val["n_tokens"]

# Dùng giới hạn đủ cho cả train và validation
MAX_LENGTH = max(
    max(train_lengths),
    max(val_lengths),
)

assert MAX_LENGTH <= model.config.max_position_embeddings, (
    f"Có mẫu dài {MAX_LENGTH} token, vượt giới hạn "
    f"{model.config.max_position_embeddings} token của model."
)

# Dữ liệu đưa vào SFTTrainer chỉ còn prompt và completion
sft_train = prepared_train.remove_columns(["n_tokens"])
sft_val = prepared_val.remove_columns(["n_tokens"])

print("Số mẫu train:", len(sft_train))
print("Số mẫu validation:", len(sft_val))
print(
    "Độ dài trung bình tập train:",
    round(sum(train_lengths) / len(train_lengths)),
    "token",
)
print("Độ dài lớn nhất tập train:", max(train_lengths), "token")
print("Độ dài lớn nhất tập validation:", max(val_lengths), "token")
print("MAX_LENGTH dùng chung:", MAX_LENGTH)
print("Các cột sau định dạng:", sft_train.column_names)