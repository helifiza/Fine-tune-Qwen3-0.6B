import re
import pandas as pd
from IPython.display import display

FILE_PATH = "/content/drive/MyDrive/VCCorp/data_luat_huggingFace_validate&test.jsonl"

audit = pd.read_json(FILE_PATH, lines=True, encoding="utf-8-sig")
audit.insert(0, "row_id", range(len(audit)))

# 1. Tách lượt user và assistant từ cột text
texts = audit["text"].fillna("").astype(str)

user_text = texts.str.extract(
    r"<\|im_start\|>user\s*\n(.*?)<\|im_end\|>",
    flags=re.DOTALL,
    expand=False,
)

assistant_text = texts.str.extract(
    r"<\|im_start\|>assistant\s*\n(.*?)<\|im_end\|>",
    flags=re.DOTALL,
    expand=False,
)

# 2. Tách câu hỏi
audit["question"] = user_text.str.extract(
    r"### Question\s*\n(.*)",
    flags=re.DOTALL,
    expand=False,
).str.strip()

# Tách đáp án cuối để kiểm tra; cột text gốc vẫn giữ nguyên
audit["answer"] = assistant_text.str.replace(
    r"(?s)^\s*<think>.*?</think>\s*",
    "",
    regex=True,
).str.strip()

# 3. Kiểm tra thiếu câu hỏi, đáp án
missing = {}

for col in ["question", "answer"]:
    missing[col] = (
        audit[col].isna()
        | audit[col].fillna("").str.strip().eq("")
    )

# 4. Tạo khóa so sánh giống cách kiểm tra trước
keys = pd.DataFrame({
    "question": audit["question"].fillna("").astype(str).str.strip(),
    "answer": audit["answer"].fillna("").astype(str).str.strip(),
})

complete = ~missing["question"] & ~missing["answer"]

# Đánh dấu tất cả dòng thuộc nhóm hỏi–đáp trùng
duplicate = (
    keys.duplicated(subset=["question", "answer"], keep=False)
    & complete
)

# Tìm câu hỏi có nhiều đáp án khác nhau
answer_counts = (
    keys.loc[complete]
    .groupby("question")["answer"]
    .nunique()
)

questions_to_review = answer_counts[answer_counts > 1].index

different_answers = (
    complete & keys["question"].isin(questions_to_review)
)

# 5. Hiển thị kết quả
checks = {
    "Thiếu câu hỏi": missing["question"],
    "Thiếu đáp án": missing["answer"],
    "Thuộc nhóm hỏi–đáp trùng": duplicate,
    "Cùng câu hỏi, khác đáp án": different_answers,
}

print("Tổng số mẫu:", len(audit))

display(pd.DataFrame([
    {"Loại cần kiểm tra": name, "Số dòng": int(mask.sum())}
    for name, mask in checks.items()
]))