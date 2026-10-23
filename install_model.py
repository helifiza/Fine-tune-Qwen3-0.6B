#tải model và chạy thử
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

messages = [
    {"role": "user", "content": "Xin chào! Hãy giải thích ngắn gọn Machine Learning là gì."}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True   # đổi thành False để tắt chế độ "suy nghĩ"
)

inputs = tokenizer([text], return_tensors="pt").to(model.device)

output_ids = model.generate(
    **inputs,
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.6,
    top_p=0.95,
    top_k=20
)[0][len(inputs.input_ids[0]):].tolist()

# Tách phần suy nghĩ (<think>...</think>) và phần trả lời
try:
    index = len(output_ids) - output_ids[::-1].index(151668)  # token </think>
except ValueError:
    index = 0

thinking = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
answer = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()

print("Suy nghĩ:", thinking)
print("Trả lời:", answer)