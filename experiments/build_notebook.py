#!/usr/bin/env python3
"""Generate experiments/mygo_kaggle.ipynb.

This script builds the Kaggle notebook that reproduces the MyGO paper
(AAAI 2025) on DB15K: the full model and the "w/o L_con" ablation.

It reads the CURRENT content of experiments/parse_log.py and
experiments/wandb_stream.py from disk and inlines each into a `%%writefile`
cell, so regenerating the notebook always stays in sync with the tested
parsers (run `python3 -m pytest -q` in this
directory before regenerating, to make sure the parser is green).

Usage:
    python3 experiments/build_notebook.py

Produces:
    experiments/mygo_kaggle.ipynb
"""

from __future__ import annotations

import os

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
PARSE_LOG_PATH = os.path.join(HERE, "parse_log.py")
WANDB_STREAM_PATH = os.path.join(HERE, "wandb_stream.py")
OUT_PATH = os.path.join(HERE, "mygo_kaggle.ipynb")


def md(src: str) -> "nbf.NotebookNode":
    return nbf.v4.new_markdown_cell(src.strip() + "\n")


def code(src: str) -> "nbf.NotebookNode":
    return nbf.v4.new_code_cell(src.strip() + "\n")


def build_cells(parse_log_source: str, wandb_stream_source: str) -> list:
    cells = []

    # ------------------------------------------------------------------
    # 0. Title / overview
    # ------------------------------------------------------------------
    cells.append(md(r"""
# Tái lập MyGO (AAAI 2025) trên DB15K

Notebook này tái lập mô hình **MyGO** (*Tokenization, Fusion, and Augmentation: Towards
Fine-grained Multi-modal Entity Representation*, AAAI 2025) trên bộ dữ liệu **DB15K**, gồm
hai cấu hình:

- **Full model**: hàm mất mát đầy đủ, bao gồm fine-grained contrastive loss `L_con` (`--mu 0.01`).
- **Ablation w/o L_con**: bỏ fine-grained contrastive loss (`--mu 0`).

Repo gốc của tác giả: <https://github.com/zjukg/MyGO> (chỉ dùng đúng dataset/hyperparameter
mà tác giả công bố trong `run.sh`, không đổi kiến trúc mô hình).

**Vì sao chạy trên Kaggle chứ không phải máy cục bộ:** trên GPU cục bộ (NVIDIA MX330, 2 GB
VRAM) mô hình bị CUDA out-of-memory ngay ở batch huấn luyện đầu tiên khi `--mu 0.01`, vì
fine-grained contrastive loss (`contrastive_loss_finegrained`) mã hoá lại **toàn bộ** 12.842
entity của DB15K qua một `TransformerEncoder` (`hidden_dim=1024`) ở **mỗi** bước huấn luyện,
không phụ thuộc `--batch_size`. Với `--mu 0` GPU bị bão hoà VRAM (swap sang RAM qua PCIe) và
không hoàn thành nổi một epoch trong thời gian hợp lý (chi tiết: mục tái lập trong báo cáo).
GPU Kaggle (T4 x2 hoặc P100, 16 GB
VRAM) có bộ nhớ gấp khoảng 8 lần MX330 nên được kỳ vọng chạy trơn tru.

**Ghi nhận bằng Weights & Biases:** mỗi lượt training là một run W&B riêng trong project
`mygo-db15k-repro` (`full_mu0.01` và `no_lcon_mu0`). Trong lúc training chạy, notebook đọc
theo file log của mã tác giả và ghi lên W&B loss từng epoch cùng các chỉ số valid/test sau
mỗi lượt đánh giá; mã tác giả không bị sửa để làm việc này.

**Cách chạy notebook này (tóm tắt, xem thêm `README.md` của repo):**
1. Bật GPU: Settings → Accelerator → **GPU T4 x2** hoặc **P100**. Bật Internet: Settings →
   Internet → **On** (lần đầu cần xác minh số điện thoại).
2. Thêm W&B API key: Add-ons → Secrets → secret tên **`WANDB_API_KEY`**, bật công tắc gắn
   vào notebook này.
3. Chỉnh `RUN` ở mục 0. Một lượt full model mất khoảng 10 giờ, nên chạy hai phiên Kaggle
   riêng: phiên 1 `RUN = "full"`, phiên 2 `RUN = "no_lcon"` (cùng `EPOCHS`).
   - 1500 epoch như bài báo KHÔNG vừa 12 giờ của một phiên Kaggle (~44 s/epoch cho full
     model), nên notebook dùng `EPOCHS = 800`. Đây là sai khác có chủ đích so với bài báo.
   - Muốn đo lại thời gian trên GPU được cấp, đặt `RUN_SMOKE = True` (mục 8 chạy 2 epoch
     mỗi cấu hình, mất thêm khoảng 10 phút).
4. Bấm **Save Version → Save & Run All (Commit)**. Kaggle sẽ chạy toàn bộ notebook trên máy
   chủ của họ ở chế độ nền, không phụ thuộc việc trình duyệt của bạn có đang mở hay bị mất
   kết nối Internet hay không (điều này quan trọng vì các lượt training kéo dài hàng giờ).
5. Theo dõi tiến độ trực tiếp trên W&B. Sau khi Commit chạy xong, tải `mygo_results.zip`
   ở tab **Output** để lưu trữ log gốc.

Ghi chú thuật ngữ: các thuật ngữ kỹ thuật tiếng Anh (MMKGC, token, contrastive learning, ...)
được giữ nguyên trong văn bản tiếng Việt bên dưới.
"""))

    # ------------------------------------------------------------------
    # 1. RUN switch
    # ------------------------------------------------------------------
    cells.append(md(r"""
## 0. Công tắc `RUN` và số epoch `EPOCHS`

Đặt các giá trị này **trước khi** bấm Save & Run All (Commit). Xem hướng dẫn chọn giá trị ở
mục 8 (cell đo thời gian) bên dưới.
"""))
    cells.append(code(r"""
# RUN quyết định (các) lượt training nào sẽ chạy trong phiên Kaggle này.
#   "both"    -> chạy cả full model (mu=0.01) và ablation w/o L_con (mu=0)
#   "full"    -> chỉ chạy full model
#   "no_lcon" -> chỉ chạy ablation w/o L_con
# Nếu cell đo thời gian ở mục 8 cho thấy cả hai lượt KHÔNG vừa trong 12 giờ, hãy tách
# thành hai phiên Kaggle: một phiên RUN="full", một phiên RUN="no_lcon".
RUN = "full"

# RUN_SMOKE: True -> mục 8 chạy thử 2 epoch mỗi cấu hình để đo thời gian (thêm ~10 phút).
# Đã đo ở lần chạy trước (~44 s/epoch cho full model), nên mặc định bỏ qua.
RUN_SMOKE = False

# EPOCHS: số epoch huấn luyện cho MỖI lượt. Bài báo/run.sh dùng 1500, nhưng đo thực tế trên
# Kaggle (~44 s/epoch cho full model) cho thấy 1500 epoch ~18.7 giờ > 12 giờ của một phiên.
# Dùng CÙNG giá trị cho full và w/o L_con để ablation công bằng; báo cáo phải ghi rõ đây là
# sai khác so với 1500 epoch của bài báo. VALID_EPOCH: chu kỳ đánh giá (mặc định tác giả = 50).
EPOCHS = 800
VALID_EPOCH = 50

assert RUN in ("both", "full", "no_lcon"), f"RUN không hợp lệ: {RUN!r}"
assert EPOCHS >= VALID_EPOCH and EPOCHS % VALID_EPOCH == 0, "EPOCHS nên là bội của VALID_EPOCH"
print("RUN =", RUN, "| RUN_SMOKE =", RUN_SMOKE, "| EPOCHS =", EPOCHS, "| VALID_EPOCH =", VALID_EPOCH)
"""))

    # ------------------------------------------------------------------
    # 1b. W&B login (fail fast, before hours of training)
    # ------------------------------------------------------------------
    cells.append(md(r"""
## 0b. Đăng nhập Weights & Biases

Đọc API key từ Kaggle Secret `WANDB_API_KEY` (ngoài Kaggle: từ biến môi trường cùng tên).
Nếu không đọc được key, cell này dừng notebook ngay, tránh chạy training hàng giờ mà không
được ghi nhận lên W&B.
"""))
    cells.append(code(r"""
try:
    import wandb
except ImportError:
    !pip install -q wandb
    import wandb

WANDB_PROJECT = "mygo-db15k-repro"

try:
    from kaggle_secrets import UserSecretsClient
    WANDB_KEY = UserSecretsClient().get_secret("WANDB_API_KEY")
except Exception as e:  # noqa: BLE001 - not on Kaggle, or secret not attached
    import os
    WANDB_KEY = os.environ.get("WANDB_API_KEY")
    if not WANDB_KEY:
        raise RuntimeError(
            "Không đọc được WANDB_API_KEY. Trên Kaggle: Add-ons > Secrets > thêm secret "
            "WANDB_API_KEY và BẬT công tắc gắn vào notebook này."
        ) from e

assert wandb.login(key=WANDB_KEY, relogin=True), "wandb.login thất bại -- kiểm tra lại API key"
print("Đã đăng nhập W&B, wandb", wandb.__version__, "| project:", WANDB_PROJECT)
"""))

    # ------------------------------------------------------------------
    # 2. nvidia-smi
    # ------------------------------------------------------------------
    cells.append(md("## 1. Kiểm tra GPU"))
    cells.append(code("!nvidia-smi"))

    # ------------------------------------------------------------------
    # 3. clone repo
    # ------------------------------------------------------------------
    cells.append(md(r"""
## 2. Tải mã nguồn MyGO

Internet trên Kaggle đôi khi không ổn định, nên bước clone có cơ chế thử lại và bỏ qua nếu
thư mục `MyGO` đã tồn tại (an toàn khi chạy lại cell này giữa chừng).
"""))
    cells.append(code(r"""
import os
import time


def _looks_like_mygo_repo(path):
    return os.path.isfile(os.path.join(path, "train_mygo_fgc.py"))


# REPO_DIR is always computed as an ABSOLUTE path, and computed the same way
# no matter where the current working directory happens to be when this cell
# runs. This matters because the cell must be safe to re-run in the same
# kernel session (e.g. after Interrupt + Run again): if REPO_DIR were the
# relative path "MyGO", re-running after a successful first run (cwd already
# .../MyGO) would find os.path.isdir("MyGO") False, clone AGAIN into a nested
# MyGO/MyGO, and every later cell would then silently operate on that
# unpatched nested clone instead of the real one.
if os.path.isdir("/kaggle/working"):
    KAGGLE_BASE = "/kaggle/working"
elif _looks_like_mygo_repo(os.getcwd()):
    # Re-running this cell while already inside the MyGO checkout: anchor to
    # its parent directory, not to the checkout itself.
    KAGGLE_BASE = os.path.dirname(os.getcwd())
else:
    KAGGLE_BASE = os.getcwd()

REPO_DIR = os.path.join(KAGGLE_BASE, "MyGO")

if os.path.isdir(REPO_DIR):
    print(f"Thư mục {REPO_DIR} đã tồn tại, bỏ qua bước clone.")
else:
    ok = False
    for attempt in range(1, 6):
        print(f"git clone --depth 1 (lần thử {attempt}/5) ...")
        ret = os.system(f"git clone --depth 1 https://github.com/zjukg/MyGO {REPO_DIR}")
        if ret == 0 and os.path.isdir(REPO_DIR):
            ok = True
            print("Clone thành công.")
            break
        print("Clone thất bại, thử lại sau 5 giây...")
        time.sleep(5)
    if not ok:
        raise RuntimeError(
            "Không thể clone MyGO sau 5 lần thử. Kiểm tra Settings > Internet = On "
            "(cần xác minh số điện thoại lần đầu dùng Internet trên Kaggle)."
        )

os.chdir(REPO_DIR)
KAGGLE_WORKING = os.path.dirname(REPO_DIR)
print("Thư mục làm việc hiện tại:", os.getcwd())
print("Thư mục /kaggle/working tương ứng:", KAGGLE_WORKING)
"""))

    # ------------------------------------------------------------------
    # 4. transformers
    # ------------------------------------------------------------------
    cells.append(md(r"""
## 3. Cài đặt `transformers` (chỉ khi chưa có)

Kaggle đã có sẵn `torch`; `transformers` cần cho bước xây dựng `tokens/textual.pth`
(embedding đầu vào của BERT).
"""))
    cells.append(code(r"""
try:
    import transformers
    print("transformers đã có sẵn, phiên bản:", transformers.__version__)
except ImportError:
    print("Chưa có transformers, đang cài đặt...")
    !pip install -q transformers
    import transformers
    print("Đã cài transformers, phiên bản:", transformers.__version__)
"""))

    # ------------------------------------------------------------------
    # 5. textual.pth
    # ------------------------------------------------------------------
    cells.append(md(r"""
## 4. Xây dựng `tokens/textual.pth` (embedding token văn bản từ BERT)

`tokens/textual.pth` là ma trận embedding đầu vào (`get_input_embeddings().weight`) của
`bert-base-uncased`, kích thước `(30522, 768)`. Đây là model **public** trên Hugging Face
Hub, **không cần** thiết lập Kaggle Secret hay đăng nhập Hugging Face. Vì mạng trên Kaggle
đôi khi không ổn định, bước tải có cơ chế thử lại.
"""))
    cells.append(code(r"""
import time

import torch
from transformers import AutoModel

os.makedirs("tokens", exist_ok=True)
TEXTUAL_PTH = "tokens/textual.pth"

if os.path.isfile(TEXTUAL_PTH):
    print(f"{TEXTUAL_PTH} đã tồn tại, bỏ qua bước tải.")
else:
    last_err = None
    for attempt in range(1, 6):
        try:
            print(f"Tải bert-base-uncased (lần thử {attempt}/5) ...")
            bert = AutoModel.from_pretrained("bert-base-uncased")
            emb = bert.get_input_embeddings().weight.detach().clone()
            assert tuple(emb.shape) == (30522, 768), f"Kích thước bất thường: {tuple(emb.shape)}"
            torch.save(emb, TEXTUAL_PTH)
            print("Đã lưu", TEXTUAL_PTH, "shape:", tuple(emb.shape))
            last_err = None
            break
        except Exception as e:  # noqa: BLE001 - retry loop, report all errors
            last_err = e
            print(f"Lỗi: {e!r}. Thử lại sau 10 giây...")
            time.sleep(10)
    if last_err is not None:
        raise RuntimeError("Không thể tải bert-base-uncased sau 5 lần thử.") from last_err
"""))

    # ------------------------------------------------------------------
    # 6. unzip visual tokens
    # ------------------------------------------------------------------
    cells.append(md("## 5. Giải nén visual token của DB15K"))
    cells.append(code(r"""
if os.path.isfile("tokens/DB15K-visual.json"):
    print("tokens/DB15K-visual.json đã tồn tại, bỏ qua giải nén.")
else:
    !cd tokens && unzip -o DB15K-visual.json.zip
    assert os.path.isfile("tokens/DB15K-visual.json"), "Giải nén thất bại: không thấy tokens/DB15K-visual.json"
    print("Đã giải nén tokens/DB15K-visual.json")
"""))

    # ------------------------------------------------------------------
    # 7. patches
    # ------------------------------------------------------------------
    cells.append(md(r"""
## 6. Áp dụng các thay đổi (patch) lên mã nguồn gốc

Chỉ hai thay đổi này được áp dụng lên mã nguồn tác giả (`model_mygo.py`), không thay đổi
kiến trúc mô hình hay hyperparameter nào:

1. **Xoá dòng debug `print(ent_embs.shape, self.lp_token.shape)`** trong
   `contrastive_loss_finegrained` (dòng này in ra ở MỖI bước huấn luyện khi `--mu != 0`,
   gây tràn ngập log). Thay bằng `pass` để giữ nguyên cấu trúc thụt lề.
2. **`torch.load(..., weights_only=False)`**: kể từ PyTorch 2.6, giá trị mặc định của
   `weights_only` đổi thành `True`, có thể khiến việc load các file `.pth` chứa tensor thuần
   (không phải checkpoint dict) bị lỗi. Patch này **chỉ áp dụng nếu `torch.__version__ >= 2.6`**
   (dò bằng Python, không hard-code); nếu Kaggle cài PyTorch cũ hơn 2.6 thì đây là no-op.

Cell dưới in ra `git diff` sau khi patch để đối chiếu.
"""))
    cells.append(code(r"""
# Patch 1: xoá debug print (thay bằng `pass`, giữ nguyên thụt lề)
!sed -i 's/^\(\s*\)print(ent_embs.shape, self.lp_token.shape)/\1pass/' model_mygo.py

assert "print(ent_embs.shape, self.lp_token.shape)" not in open("model_mygo.py").read(), (
    "Patch 1 thất bại: debug print vẫn còn trong model_mygo.py"
)
print("Patch 1 OK: đã xoá debug print trong contrastive_loss_finegrained")
"""))
    cells.append(code(r"""
# Patch 2: torch.load(..., weights_only=False), chỉ khi torch >= 2.6 (no-op nếu không)
import re

MODEL_PATH = "model_mygo.py"
with open(MODEL_PATH) as f:
    src = f.read()

major, minor = (int(x) for x in torch.__version__.split("+")[0].split(".")[:2])

if (major, minor) < (2, 6):
    print(f"torch {torch.__version__} < 2.6: weights_only mặc định vẫn là False -> không cần patch (no-op).")
elif "weights_only=False" in src:
    print("Patch 2 đã được áp dụng từ trước (rerun cell), bỏ qua.")
else:
    patched, n = re.subn(
        r'torch\.load\((\s*"[^"]+"\s*)\)',
        r"torch.load(\1, weights_only=False)",
        src,
    )
    assert n == 5, f"Kỳ vọng vá 5 lệnh torch.load(...), thực tế vá {n} -- kiểm tra lại model_mygo.py"
    with open(MODEL_PATH, "w") as f:
        f.write(patched)
    print(f"Patch 2 OK: đã thêm weights_only=False cho {n} lệnh torch.load(...) (torch {torch.__version__} >= 2.6)")
"""))
    cells.append(code(r"""
!git diff -- model_mygo.py
"""))

    # ------------------------------------------------------------------
    # 8. parser inline
    # ------------------------------------------------------------------
    cells.append(md(r"""
## 7. Công cụ phân tích log huấn luyện (`parse_log.py`)

`train_mygo_fgc.py` (mã tác giả) ghi log huấn luyện vào
`./logs/<exp>/DB15K/<tên_dài_theo_hyperparameter>.log`, đồng thời in ra stdout. Vì stdout
đầy đủ rất dài (thanh tiến trình `tqdm`, các dòng tensor debug khác), các cell training bên
dưới sẽ chuyển hướng (`>`) toàn bộ stdout/stderr vào file rồi chỉ hiển thị `tail`.

Cell dưới nạp (`%%writefile`) đúng nội dung hiện tại của `experiments/parse_log.py` trong
repo (đã có bộ test riêng `experiments/test_parse_log.py`,
bao gồm cả trường hợp log chưa có lượt đánh giá nào, đúng tình huống từng gặp khi chạy thử
cục bộ trên GPU 2 GB).
"""))
    writefile_src = "%%writefile parse_log.py\n" + parse_log_source
    parser_cell = nbf.v4.new_code_cell(writefile_src)
    cells.append(parser_cell)
    cells.append(md(r"""
`wandb_stream.py` đọc theo file log trong lúc training đang chạy (giống `tail -f`) và biến
mỗi dòng hoàn chỉnh thành bản ghi gửi lên W&B: loss sau mỗi epoch, MRR/Hit@k (đơn vị %) và
MR sau mỗi lượt đánh giá valid/test. Nội dung giống hệt `experiments/wandb_stream.py`
(có bộ test riêng `test_wandb_stream.py`).
"""))
    cells.append(nbf.v4.new_code_cell("%%writefile wandb_stream.py\n" + wandb_stream_source))
    cells.append(code(r"""
from parse_log import parse_log, summarize, epoch_times
from wandb_stream import LogStreamParser, follow_log

print("Đã nạp parse_log.py và wandb_stream.py")
"""))

    # ------------------------------------------------------------------
    # 9. smoke/timing
    # ------------------------------------------------------------------
    cells.append(md(r"""
## 8. Đo thời gian thực tế trên GPU Kaggle (smoke test 2 epoch, cả hai cấu hình)

Chỉ chạy khi `RUN_SMOKE = True`; smoke test không được ghi lên W&B.

Chạy 2 epoch thật (`--num_epoch 2 --valid_epoch 2`) cho **từng** cấu hình, full model
(`--mu 0.01`) và w/o L_con (`--mu 0`), để đo (không giả định `mu=0` nhanh hơn, vì chưa
từng được đo trên GPU đủ lớn):

- **giây/epoch** (từ nhãn thời gian của 2 dòng epoch trong log),
- **thời gian một lượt đánh giá** (valid + test), từ nhãn thời gian dòng
  "Link Prediction on Validation Set" đến dòng `Hit1` cuối cùng của Test Set,

rồi suy ra ước tính cho **`EPOCHS` epoch + `EPOCHS / VALID_EPOCH` lượt đánh giá** của mỗi
cấu hình và so với ngân sách 12 giờ của một phiên Kaggle. (Với 1500 epoch, full model
~18.7 giờ, không vừa một phiên; đó là lý do dùng `EPOCHS` nhỏ hơn.)
"""))
    cells.append(code(r"""
import datetime
import glob
import re
import shutil

TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")
EPOCH_RE = re.compile(r"-\s(\d+)\s*\t\s*([\d.]+)\s*\t\s*([\d.]+) s")


def parse_ts(line: str):
    m = TS_RE.match(line)
    if not m:
        return None
    return datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f")


def run_smoke(exp: str, mu: str):
    # Chạy 2 epoch thật với mu cho trước; trả về (giây/epoch, giây/lượt đánh giá).
    shutil.rmtree(f"logs/{exp}", ignore_errors=True)
    cmd = (
        "python train_mygo_fgc.py --data DB15K --num_epoch 2 --valid_epoch 2 "
        "--hidden_dim 1024 --lr 1e-3 --dim 256 --max_vis_token 8 --max_txt_token 4 "
        "--num_head 2 --emb_dropout 0.6 --vis_dropout 0.3 --txt_dropout 0.1 "
        f"--num_layer_dec 1 --mu {mu} --exp {exp} > run_{exp}.out 2>&1"
    )
    print("Đang chạy:", cmd)
    ret = os.system(cmd)
    print("Exit code:", ret)
    os.system(f"tail -n 15 run_{exp}.out")

    logs = glob.glob(f"logs/{exp}/DB15K/*.log")
    assert logs, (
        f"Không tìm thấy log {exp} -- xem run_{exp}.out để biết lỗi "
        "(ví dụ CUDA OOM hoặc thiếu file token)."
    )
    print("Log smoke:", logs[0])

    epoch_ts, valid_start_ts, test_end_ts = [], None, None
    for line in open(logs[0], encoding="utf-8"):
        ts = parse_ts(line)
        if ts is None:
            continue
        if EPOCH_RE.search(line):
            epoch_ts.append(ts)
        if "Link Prediction on Validation Set" in line:
            valid_start_ts = ts
        if "Hit1:" in line:
            # Ghi đè: dòng Hit1 cuối cùng trong log thuộc Test Set (chạy sau Validation Set).
            test_end_ts = ts

    assert len(epoch_ts) >= 2, (
        f"Cần ít nhất 2 dòng epoch để đo giây/epoch, log {exp} chỉ có {len(epoch_ts)} dòng "
        f"(xem run_{exp}.out)."
    )
    assert valid_start_ts is not None and test_end_ts is not None, (
        f"Không tìm thấy đủ dấu thời gian của lượt đánh giá (valid/test) trong log {exp}."
    )
    return (epoch_ts[1] - epoch_ts[0]).total_seconds(), (test_end_ts - valid_start_ts).total_seconds()


if not RUN_SMOKE:
    print("RUN_SMOKE = False -> bỏ qua smoke test (thời gian đã đo ở lần chạy trước: ~44 s/epoch full model).")
else:
    SMOKE = {
        "full": run_smoke("kaggle_smoke_full", "0.01"),
        "no_lcon": run_smoke("kaggle_smoke_no_lcon", "0"),
    }

    N_EVALS = EPOCHS // VALID_EPOCH
    KAGGLE_BUDGET_SECONDS = 12 * 3600

    est = {}
    for name, (sec_per_epoch, eval_sec) in SMOKE.items():
        est[name] = sec_per_epoch * EPOCHS + eval_sec * N_EVALS
        est_1500 = sec_per_epoch * 1500 + eval_sec * 30
        print(f"[{name}] {sec_per_epoch:.2f} s/epoch, {eval_sec:.2f} s/lượt đánh giá -> "
              f"{est[name]/3600:.2f} giờ cho {EPOCHS} epoch + {N_EVALS} lượt đánh giá "
              f"(tham chiếu 1500 epoch: {est_1500/3600:.2f} giờ)")

    selected = ["full", "no_lcon"] if RUN == "both" else [RUN]
    est_seconds_total = sum(est[n] for n in selected)
    print(f"RUN = {RUN!r}, EPOCHS = {EPOCHS} -> các lượt trong phiên này: {selected}")
    print(f"Ước tính tổng thời gian phiên này: {est_seconds_total/3600:.2f} giờ "
          f"(ngân sách Kaggle: {KAGGLE_BUDGET_SECONDS/3600:.0f} giờ)")

    if est_seconds_total < 0.85 * KAGGLE_BUDGET_SECONDS:
        print("=> OK: ước tính vừa trong ngân sách 12 giờ (còn dư ~15% an toàn).")
    else:
        print(
            "=> CẢNH BÁO: ước tính SÁT hoặc VƯỢT 12 giờ. Giảm EPOCHS, hoặc đặt RUN thành "
            "\"full\" cho phiên này và chạy \"no_lcon\" ở một phiên Kaggle Commit riêng "
            "(cùng EPOCHS)."
        )
"""))

    # ------------------------------------------------------------------
    # 10. training with live W&B logging
    # ------------------------------------------------------------------
    cells.append(md(r"""
## 9. Hàm chạy training và ghi nhận lên W&B

Mỗi lượt training được chạy bằng `subprocess.Popen`, toàn bộ stdout/stderr (kể cả thanh tiến
trình `tqdm`) ghi vào `run_<exp>.out` thay vì in ra notebook. Cách này, kết hợp với
**Save & Run All (Commit)**, đảm bảo lượt training hàng giờ vẫn chạy tiếp trên máy chủ Kaggle
kể cả khi trình duyệt của bạn bị mất kết nối hoặc đóng lại.

Trong lúc tiến trình chạy, `follow_log` đọc theo file log của mã tác giả và gửi lên W&B:

- `train/loss`, `train/epoch_time_s` sau mỗi epoch;
- `valid/*`, `test/*` (MRR, Hit@1/3/10 theo %, MR) sau mỗi `VALID_EPOCH` epoch;
- trục hoành của mọi biểu đồ là `epoch`.

Khi training xong: `summary` của run ghi kết quả chọn theo valid (`best_by_valid/*`) và theo
test (`best_by_test/*`), file log gốc được tải lên dưới dạng artifact. Toàn bộ hyperparameter
nằm trong `config` của run.

Nếu cell chạy lại trong cùng một phiên, log cũ của thí nghiệm đó bị xoá và huấn luyện lại từ
đầu (tạo một run W&B mới), để tránh log bị nối chồng.
"""))
    cells.append(code(r"""
import subprocess
import sys

# Hyperparameter của tác giả cho DB15K (run.sh), dùng chung cho cả hai cấu hình; chỉ mu khác.
TRAIN_ARGS = {
    "data": "DB15K", "num_epoch": EPOCHS, "valid_epoch": VALID_EPOCH,
    "hidden_dim": 1024, "lr": 1e-3, "dim": 256, "max_vis_token": 8, "max_txt_token": 4,
    "num_head": 2, "emb_dropout": 0.6, "vis_dropout": 0.3, "txt_dropout": 0.1,
    "num_layer_dec": 1,
}
RUN_URLS = {}


def run_with_wandb(exp: str, mu: str, run_name: str) -> int:
    shutil.rmtree(f"logs/{exp}", ignore_errors=True)
    args = dict(TRAIN_ARGS, mu=mu, exp=exp)
    cmd = [sys.executable, "train_mygo_fgc.py"]
    for k, v in args.items():
        cmd += [f"--{k}", str(v)]
    print("Lệnh:", " ".join(cmd))

    run = wandb.init(
        project=WANDB_PROJECT, name=run_name, group="DB15K", job_type="train",
        config=dict(args, batch_size=2048, paper_num_epoch=1500,
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
    )
    run.define_metric("epoch")
    run.define_metric("*", step_metric="epoch")
    RUN_URLS[exp] = run.url
    print("W&B run:", run.url)

    parser = LogStreamParser()

    def on_line(line):
        for event in parser.feed(line):
            run.log(event)

    with open(f"run_{exp}.out", "w") as out:
        proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT)
        log_path = follow_log(f"logs/{exp}/DB15K/*.log", proc, on_line, poll_s=10)
    ret = proc.wait()
    print("Exit code:", ret)

    if log_path is not None:
        s = summarize(parse_log(log_path))
        for kind in ("best_by_valid", "best_by_test"):
            if s[kind] is not None:
                run.summary[f"{kind}/epoch"] = s[kind]["epoch"]
                for key, name in (("MRR", "MRR"), ("Hit1", "Hit@1"), ("Hit3", "Hit@3"), ("Hit10", "Hit@10")):
                    run.summary[f"{kind}/{name}"] = s[kind][key] * 100
        art = wandb.Artifact(f"{exp}-training-log", type="training-log")
        art.add_file(log_path)
        run.log_artifact(art)
    else:
        print(f"Không thấy log của {exp} -- xem run_{exp}.out")
    run.summary["exit_code"] = ret
    run.finish(exit_code=ret)
    return ret
"""))

    cells.append(md(r"""
## 10. Chạy Full model (`--mu 0.01`, `EPOCHS` epoch)
"""))
    cells.append(code(r"""
if RUN in ("both", "full"):
    print("Đang chạy FULL model (mu=0.01) -- khoảng 10 giờ với 800 epoch.")
    run_with_wandb("full", "0.01", "full_mu0.01")
else:
    print(f"RUN={RUN!r} -> bỏ qua lượt full model trong phiên này.")
"""))
    cells.append(code(r"""
if os.path.isfile("run_full.out"):
    !tail -n 60 run_full.out
else:
    print("run_full.out chưa tồn tại (lượt full chưa chạy trong phiên này).")
"""))

    cells.append(md(r"""
## 10b. Chạy Ablation w/o L_con (`--mu 0`, `EPOCHS` epoch)

Giống hệt full model, chỉ khác `--mu 0` (tắt fine-grained contrastive loss `L_con`).
"""))
    cells.append(code(r"""
if RUN in ("both", "no_lcon"):
    print("Đang chạy ablation w/o L_con (mu=0).")
    run_with_wandb("no_lcon", "0", "no_lcon_mu0")
else:
    print(f"RUN={RUN!r} -> bỏ qua lượt w/o L_con trong phiên này.")
"""))
    cells.append(code(r"""
if os.path.isfile("run_no_lcon.out"):
    !tail -n 60 run_no_lcon.out
else:
    print("run_no_lcon.out chưa tồn tại (lượt w/o L_con chưa chạy trong phiên này).")
"""))

    # ------------------------------------------------------------------
    # 12. summarize
    # ------------------------------------------------------------------
    cells.append(md(r"""
## 11. Tổng hợp kết quả và so sánh với bài báo

Log ghi các chỉ số MRR/Hit@1/Hit@3/Hit@10 dưới dạng phân số (ví dụ `0.3772`); bảng dưới hiển
thị dưới dạng phần trăm với 2 chữ số thập phân.

Hai cách chọn "best" khác nhau đều được hiển thị:
- **best_by_valid**: chọn epoch có validation MRR cao nhất (model selection đúng chuẩn), rồi
  báo cáo chỉ số TEST của epoch đó.
- **best_by_test**: chọn theo test MRR cao nhất, đây là cách mã gốc `train_mygo_fgc.py` tự
  theo dõi biến `best_mrr`/`best_result` trong lúc huấn luyện.

Nếu một lượt chưa có log hoàn chỉnh nào (`RUN` không bao gồm lượt đó trong phiên này, hoặc
lượt đó bị dừng trước lượt đánh giá đầu tiên), bảng sẽ hiển thị "chưa có lượt đánh giá" thay
vì báo lỗi.
"""))
    cells.append(code(r"""
import glob

import pandas as pd

PAPER = {
    "full": {"MRR": 37.72, "Hit1": 30.08, "Hit3": 41.26, "Hit10": 52.21},
    "no_lcon": {"MRR": 35.99, "Hit1": 27.98, "Hit3": 39.68, "Hit10": 51.31},
}
EXPS = [("full", "Full model"), ("no_lcon", "w/o L_con (mu=0)")]


def find_log(exp: str):
    matches = sorted(glob.glob(f"logs/{exp}/DB15K/*.log"))
    return matches[0] if matches else None


table_rows = []
plot_series = {}

for exp, label in EXPS:
    log_path = find_log(exp)
    if log_path is None:
        table_rows.append({"Model": label, "Best-by": "-", "Epoch": "-",
                            "MRR (%)": "chưa có lượt đánh giá", "Hit@1 (%)": "-",
                            "Hit@3 (%)": "-", "Hit@10 (%)": "-"})
    else:
        rows = parse_log(log_path)
        plot_series[label] = [(r["epoch"], r["MRR"] * 100) for r in rows if r["split"] == "valid"]
        s = summarize(rows)

        for kind in ("best_by_valid", "best_by_test"):
            m = s[kind]
            if m is None:
                table_rows.append({"Model": label, "Best-by": kind, "Epoch": "-",
                                    "MRR (%)": "chưa có lượt đánh giá", "Hit@1 (%)": "-",
                                    "Hit@3 (%)": "-", "Hit@10 (%)": "-"})
            else:
                table_rows.append({
                    "Model": label, "Best-by": kind, "Epoch": m["epoch"],
                    "MRR (%)": f"{m['MRR'] * 100:.2f}",
                    "Hit@1 (%)": f"{m['Hit1'] * 100:.2f}",
                    "Hit@3 (%)": f"{m['Hit3'] * 100:.2f}",
                    "Hit@10 (%)": f"{m['Hit10'] * 100:.2f}",
                })

    # Always show the paper's reference row for this experiment, even when its
    # log is missing (e.g. a split-session run where only "full" or only
    # "no_lcon" was executed in this Kaggle Commit), so the table still lets
    # the student compare against Table 1/2 immediately.
    p = PAPER[exp]
    table_rows.append({"Model": label, "Best-by": "paper (Table 1/2)", "Epoch": "-",
                        "MRR (%)": f"{p['MRR']:.2f}", "Hit@1 (%)": f"{p['Hit1']:.2f}",
                        "Hit@3 (%)": f"{p['Hit3']:.2f}", "Hit@10 (%)": f"{p['Hit10']:.2f}"})

df = pd.DataFrame(table_rows)
print(df.to_string(index=False))

RESULTS_DIR = os.path.join(KAGGLE_WORKING, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
csv_path = os.path.join(RESULTS_DIR, "summary.csv")
df.to_csv(csv_path, index=False)
print("Đã lưu bảng tổng hợp:", csv_path)
"""))
    cells.append(code(r"""
import matplotlib.pyplot as plt

if plot_series:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"Full model": "#2E5EAA", "w/o L_con (mu=0)": "#D9782D"}
    for label, points in plot_series.items():
        if not points:
            continue
        points = sorted(points)
        xs, ys = zip(*points)
        ax.plot(xs, ys, marker="o", label=label, color=colors.get(label))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation MRR (%)")
    ax.set_title("MyGO trên DB15K: Validation MRR theo epoch")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "valid_mrr_vs_epoch.png")
    fig.savefig(plot_path, dpi=150)
    print("Đã lưu biểu đồ:", plot_path)
else:
    print("Chưa có dữ liệu validation MRR để vẽ (chưa lượt nào có eval log trong phiên này).")
"""))

    # ------------------------------------------------------------------
    # 13. zip
    # ------------------------------------------------------------------
    cells.append(md(r"""
## 12. Đóng gói kết quả để tải về

Nén `logs/` (toàn bộ log huấn luyện, cả `full` và `no_lcon` nếu có) và `results/`
(bảng CSV + biểu đồ ở mục 11) thành `mygo_results.zip` trong `/kaggle/working`.
"""))
    cells.append(code(r"""
import zipfile

ZIP_PATH = os.path.join(KAGGLE_WORKING, "mygo_results.zip")

with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
    if os.path.isdir("logs"):
        for dirpath, _, filenames in os.walk("logs"):
            for fn in filenames:
                full = os.path.join(dirpath, fn)
                zf.write(full, arcname=full)
    if os.path.isdir(RESULTS_DIR):
        for dirpath, _, filenames in os.walk(RESULTS_DIR):
            for fn in filenames:
                full = os.path.join(dirpath, fn)
                arcname = os.path.relpath(full, KAGGLE_WORKING)
                zf.write(full, arcname=arcname)

print("Đã tạo:", ZIP_PATH)
for exp, url in RUN_URLS.items():
    print(f"W&B run {exp}: {url}")
"""))
    cells.append(md(r"""
## Hoàn tất

Link các run W&B được in ở cell trên (và có trong project `mygo-db15k-repro` trên W&B).
Để người đọc báo cáo xem được, đặt project ở chế độ **Public** (Project → Settings →
Visibility).

Vào tab **Output** của phiên bản notebook này (sau khi Commit chạy xong), tải file
`mygo_results.zip` về máy để lưu trữ log gốc. Nếu chạy tách hai phiên (`RUN="full"` và
`RUN="no_lcon"`), mỗi phiên có một file `mygo_results.zip` riêng.
"""))

    return cells


def main() -> None:
    with open(PARSE_LOG_PATH, encoding="utf-8") as f:
        parse_log_source = f.read()
    with open(WANDB_STREAM_PATH, encoding="utf-8") as f:
        wandb_stream_source = f.read()

    nb = nbf.v4.new_notebook()
    nb["cells"] = build_cells(parse_log_source, wandb_stream_source)
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    }

    nbf.validate(nb)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"Wrote {OUT_PATH} ({len(nb['cells'])} cells), validated OK.")


if __name__ == "__main__":
    main()
