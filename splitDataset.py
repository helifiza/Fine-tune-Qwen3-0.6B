import random
import re
import unicodedata
from collections import defaultdict
from datasets import load_dataset

# Đọc toàn bộ dataset mới
data = load_dataset(
    "json",
    data_files="/content/drive/MyDrive/VCCorp/data_luat_huggingFace_validate&test.jsonl",
    split="train",
)

def get_question(sample):
    if "question" in sample:
        return sample["question"].strip()

    user_text = sample["text"].split("<|im_start|>user\n", 1)[1]
    user_text = user_text.split("<|im_end|>", 1)[0]
    return user_text.split("### Question", 1)[1].strip()

def normalize_question(question):
    question = unicodedata.normalize("NFC", question).casefold()
    return " ".join(re.findall(r"\w+", question))

# Gom các dòng có cùng câu hỏi thành một nhóm
groups = defaultdict(list)

for index, sample in enumerate(data):
    key = normalize_question(get_question(sample))
    groups[key].append(index)

# Xác định số mẫu cho từng tập
total = len(data)

targets = {
    "train": round(total * 0.70),
    "validation": round(total * 0.10),
}
targets["test"] = total - targets["train"] - targets["validation"]

# Xáo trộn nhóm; xếp nhóm lớn trước để không phải tách nhóm
rng = random.Random(42)
group_list = list(groups.items())
rng.shuffle(group_list)
group_list.sort(key=lambda item: len(item[1]), reverse=True)

split_indices = {name: [] for name in targets}
split_questions = {name: set() for name in targets}

for question_key, indices in group_list:
    available = [
        name for name in targets
        if len(split_indices[name]) + len(indices) <= targets[name]
    ]

    if not available:
        raise ValueError("Không thể chia đúng số mẫu mà vẫn giữ nguyên nhóm.")

    # Đưa cả nhóm vào tập còn thiếu nhiều mẫu nhất
    chosen = max(
        available,
        key=lambda name: targets[name] - len(split_indices[name]),
    )

    split_indices[chosen].extend(indices)
    split_questions[chosen].add(question_key)

# Kiểm tra không trùng câu hỏi giữa ba tập
assert split_questions["train"].isdisjoint(split_questions["validation"])
assert split_questions["train"].isdisjoint(split_questions["test"])
assert split_questions["validation"].isdisjoint(split_questions["test"])

for name in split_indices:
    rng.shuffle(split_indices[name])
    assert len(split_indices[name]) == targets[name]

# Tạo ba tập để sử dụng ở các bước tiếp theo
train_raw = data.select(split_indices["train"])
validation_raw = data.select(split_indices["validation"])
test_raw = data.select(split_indices["test"])

print("Train:", len(train_raw))
print("Validation:", len(validation_raw))
print("Test:", len(test_raw))
print("Câu hỏi trùng giữa các tập sau chuẩn hóa: 0")