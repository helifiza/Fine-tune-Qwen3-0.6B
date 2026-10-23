# Fine-tune-Qwen3-0.6B

Mã nguồn fine-tune mô hình `Qwen/Qwen3-0.6B` bằng LoRA cho bài toán hỏi đáp pháp luật bằng tiếng Việt, kèm kiểm tra dữ liệu và so sánh mô hình trước/sau huấn luyện.

## Chức năng các file

| File | Chức năng |
| --- | --- |
| `README.md` | Hướng dẫn cài đặt thư viện, chuẩn bị môi trường và chạy các bước trong dự án. |
| `install_model.py` | Tải mô hình và tokenizer Qwen3-0.6B từ Hugging Face rồi sinh câu trả lời thử, tách riêng phần suy nghĩ và đáp án. |
| `kiemTra_dataset.py` | Kiểm tra dữ liệu JSONL để thống kê mẫu thiếu câu hỏi hoặc đáp án, cặp hỏi–đáp trùng và câu hỏi có nhiều đáp án khác nhau. |
| `splitDataset.py` | Chia dữ liệu thành train/validation/test theo tỷ lệ 70%/10%/20%, giữ các câu hỏi giống nhau sau chuẩn hóa trong cùng một tập. |
| `preProcessing.py` | Chuyển dữ liệu train và validation thành `prompt`/`completion`, loại phần suy nghĩ khỏi đáp án và thống kê độ dài token. |
| `attachLoRA_Qwen.py` | Gắn adapter LoRA vào các lớp attention của Qwen và cấu hình để chỉ huấn luyện trọng số LoRA trên GPU. |
| `runTraining.py` | Huấn luyện mô hình bằng `SFTTrainer`, đánh giá định kỳ và lưu checkpoint cùng nhật ký loss, thời gian và bộ nhớ GPU. |
| `prepareModel_compare.py` | Chuẩn bị mô hình LoRA từ phiên huấn luyện hoặc adapter đã lưu và mã hóa tập validation để so sánh. |
| `compareModel.py` | So sánh loss và perplexity trên tập validation khi tắt/bật LoRA rồi xuất kết quả từng mẫu và chỉ số tổng hợp ra CSV. |

## Cài đặt thư viện

Mã nguồn hiện dùng đường dẫn `/content/...` và yêu cầu CUDA ở bước huấn luyện, vì vậy hướng dẫn dưới đây dành cho Google Colab với GPU đã bật.

Chạy các lệnh bạn đã cung cấp trong một ô notebook:

```python
!pip install -U transformers accelerate
%pip uninstall -y torchao
!pip install -U trl
```

Cài thêm các thư viện được sử dụng trong mã nguồn, bao gồm TensorBoard để ghi log huấn luyện:

```python
%pip install torch peft datasets numpy pandas matplotlib tqdm tensorboard ipython
```

Nếu notebook yêu cầu khởi động lại phiên sau khi cài thư viện, hãy khởi động lại trước khi chạy các bước tiếp theo.

## Khởi tạo và chạy mã nguồn

1. Bật GPU cho notebook Colab, tải các file `.py` trong thư mục này vào `/content` và chọn thư mục làm việc:

   ```python
   %cd /content
   ```

2. Kết nối Google Drive chứa dữ liệu:

   ```python
   from google.colab import drive
   drive.mount("/content/drive")
   ```

   Đường dẫn dữ liệu mặc định trong `kiemTra_dataset.py` và `splitDataset.py` là `/content/drive/MyDrive/VCCorp/data_luat_huggingFace_validate&test.jsonl`; sửa cả hai file nếu dữ liệu nằm ở vị trí khác.

   Bước kiểm tra dữ liệu yêu cầu JSONL có cột `text` chứa các lượt `user` và `assistant` theo định dạng `<|im_start|>...<|im_end|>`, với câu hỏi nằm sau `### Question` trong lượt `user`.

3. Chạy lần lượt các lệnh dưới đây trong cùng một phiên notebook:

   ```python
   %run -i install_model.py
   %run -i kiemTra_dataset.py
   %run -i splitDataset.py
   %run -i preProcessing.py
   %run -i attachLoRA_Qwen.py

   # runTraining.py dùng lengths để tính phân vị độ dài token.
   lengths = train_lengths
   %run -i runTraining.py

   %run -i prepareModel_compare.py
   %run -i compareModel.py
   ```

   Các file dùng chung biến như `model`, `tokenizer`, `sft_train` và `sft_val`; `%run -i` cho phép dùng các biến của phiên notebook khi thực thi file, nên không chạy từng file bằng các lệnh `python file.py` riêng biệt.

   Chạy `preProcessing.py` trước `attachLoRA_Qwen.py` để có hàm `set_seed`, và chỉ gắn LoRA một lần trên mỗi mô hình gốc đã tải.

4. Để lưu adapter và tokenizer cho lần sử dụng sau, chạy sau khi huấn luyện:

   ```python
   adapter_dir = "/content/qwen3_luat_lora_adapter"
   trainer.model.save_pretrained(adapter_dir)
   tokenizer.save_pretrained(adapter_dir)
   ```

   Sao lưu thư mục này về máy hoặc Google Drive trước khi kết thúc phiên Colab; khi mở phiên mới, khôi phục đúng đường dẫn trên và chạy lại các bước tải mô hình, chia dữ liệu, tiền xử lý trước khi chạy hai file so sánh, dùng cùng dữ liệu đầu vào và seed để tái tạo tập validation.

## Kết quả đầu ra

| Đường dẫn trên Colab | Nội dung |
| --- | --- |
| `/content/qwen3_luat_checkpoints/` | Checkpoint và log huấn luyện, gồm `train_log.jsonl`. |
| `/content/qwen3_luat_lora_adapter/` | Adapter và tokenizer nếu chạy bước lưu thủ công. |
| `/content/so_sanh_tung_cau.csv` | Loss trước/sau fine-tune và mức giảm loss của từng mẫu validation. |
| `/content/so_sanh_metrics.csv` | Loss tổng hợp theo số token đáp án và perplexity của hai cấu hình mô hình. |

Tập test được tạo trong bước chia dữ liệu nhưng chưa được sử dụng trong quy trình đánh giá hiện tại.
