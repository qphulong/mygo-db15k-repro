# Tái lập MyGO trên DB15K

Mã nguồn chạy lại thực nghiệm của báo cáo cuối kỳ môn Máy học với dữ liệu đồ thị (MTH084):
hai cấu hình của MyGO ([Zhang và cộng sự, AAAI 2025](https://arxiv.org/abs/2404.09468)) trên
bộ dữ liệu DB15K, dùng mã gốc của tác giả <https://github.com/zjukg/MyGO>. Kết quả và phân
tích nằm trong báo cáo, không nằm trong repo này.

| Cấu hình | Tham số khác nhau | Run W&B |
|---|---|---|
| Mô hình đầy đủ | `--mu 0.01` | `full_mu0.01` |
| Không có hàm mất mát tương phản | `--mu 0` | `no_lcon_mu0` |

Quá trình chạy được ghi nhận trên Weights & Biases, project
[`mygo-db15k-repro`](https://wandb.ai/qphulong-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/mygo-db15k-repro).

## Cấu trúc

| File | Vai trò |
|---|---|
| `experiments/mygo_kaggle.ipynb` | Notebook chạy trên Kaggle: tải mã gốc và dữ liệu, chuẩn bị token, huấn luyện, ghi lên W&B, tổng hợp và đóng gói log. **Đây là file duy nhất cần chạy.** |
| `experiments/build_notebook.py` | Sinh lại notebook trên (nhúng nội dung hai file `.py` bên dưới vào notebook). |
| `experiments/parse_log.py` | Đọc file log của mã gốc, chọn epoch tốt nhất theo valid và theo test. Có thể chạy độc lập: `python3 experiments/parse_log.py <file.log>`. |
| `experiments/wandb_stream.py` | Đọc theo file log trong lúc huấn luyện và chuyển từng dòng thành bản ghi gửi lên W&B. |
| `experiments/test_*.py`, `experiments/fixtures/` | Bộ kiểm thử `pytest` cho hai công cụ trên, chạy trên log mẫu. |

## Môi trường và phần cứng

- **Nền tảng:** Kaggle Notebook, GPU **Tesla T4** 16 GB (lựa chọn "GPU T4 x2"; mã gốc chỉ dùng
  một GPU), giới hạn 12 giờ mỗi phiên.
- **Phần mềm** (ảnh Kaggle tại thời điểm chạy, tháng 10/2026): Python 3, PyTorch 2.10
  (CUDA 12.8), `transformers` 5.0, `wandb` 0.26.1. Notebook tự cài `transformers` và `wandb`
  nếu thiếu.
- **Mạng:** cần bật Internet để clone mã gốc, tải `bert-base-uncased` từ Hugging Face và gửi số
  liệu lên W&B.
- Máy có GPU 2 GB không chạy được mã gốc (hết bộ nhớ ngay batch đầu tiên); cần GPU khoảng 16 GB.

Notebook áp dụng đúng hai thay đổi lên mã gốc, in ra `git diff` để đối chiếu: bỏ một lệnh
`print` gỡ lỗi chạy ở mọi bước, và thêm `weights_only=False` cho `torch.load` (chỉ khi PyTorch
≥ 2.6). Siêu tham số lấy theo `run.sh` của tác giả cho DB15K, trừ số epoch (800 thay vì 1500,
để vừa một phiên Kaggle).

## Cách chạy lại

1. Tạo tài khoản W&B, lấy API key tại <https://wandb.ai/authorize>.
2. Trên Kaggle: **Code → New Notebook → File → Import Notebook**, chọn
   `experiments/mygo_kaggle.ipynb`.
3. Panel **Settings**: Accelerator = **GPU T4 x2** (hoặc P100), Internet = **On**.
4. **Add-ons → Secrets → Add a new secret**: label `WANDB_API_KEY`, value là API key; bật công
   tắc gắn secret vào notebook.
5. Ở ô đầu tiên của notebook, đặt `RUN = "full"` (mô hình đầy đủ, khoảng 10 giờ) và bấm
   **Save Version → Save & Run All (Commit)**.
6. Lặp lại bước 5 với `RUN = "no_lcon"` (khoảng 5 giờ). Hai phiên có thể chạy song song nếu hạn
   mức GPU cho phép (Kaggle cấp khoảng 30 giờ GPU mỗi tuần).
7. Theo dõi trên W&B, project `mygo-db15k-repro`. Khi Commit xong, tab **Output** của từng phiên
   có `mygo_results.zip` (log gốc, bảng tổng hợp, biểu đồ).

Các tùy chọn khác ở ô đầu tiên: `EPOCHS` (mặc định 800, giống nhau cho hai cấu hình),
`VALID_EPOCH` (mặc định 50), `RUN_SMOKE = True` để chạy thử 2 epoch mỗi cấu hình và ước tính
thời gian trước khi chạy thật.

Đọc lại kết quả từ log đã tải về (chạy được trên máy bất kỳ có Python 3, không cần GPU):

```bash
unzip mygo_results.zip -d mygo_results
python3 experiments/parse_log.py mygo_results/logs/*/DB15K/*.log
```

## Kiểm thử và sinh lại notebook (trên máy cục bộ)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
cd experiments
python -m pytest -q          # kiểm thử parse_log.py và wandb_stream.py
python build_notebook.py     # sinh lại mygo_kaggle.ipynb sau khi sửa các file .py
```
