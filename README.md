# Autoss Tech Sales Agent (Tokinarc) – Demo

Hệ thống demo Sales Assistant B2B cho phụ kiện MIG/MAG Tokinarc, chạy FastAPI + Gemini API + pipeline theo ADK, có UI đơn giản và thinking logs hiển thị.

## Cài đặt & chạy

### 1) Tạo venv + cài dependencies

PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

CMD:
```bat
py -m venv .venv
.\.venv\Scripts\activate
pip install -r backend\requirements.txt
```

### 2) Cấu hình môi trường

Tạo `backend/.env`:
```
GEMINI_API_KEY=YOUR_KEY_HERE
GEMINI_MODEL_FLASH=gemini-2.5-flash
GEMINI_MODEL_PRO=gemini-2.5-pro
MAX_ATTEMPTS=3
MAX_IMAGES=4
```

### 3) Chạy API

```powershell
uvicorn backend.app:app --reload
```

### 4) Mở UI

Mở `http://127.0.0.1:8000`.

## Nguồn dữ liệu

- Dữ liệu duy nhất: `resources/AgentX.json` (đọc mỗi request).
- Chỉ hiển thị thông tin có trong file.
- Link ảnh lấy từ trường `Link sản phẩm` (nếu có).
- Mã hoá file khuyến nghị: UTF-8.

## Nguyên lý vận hành (theo từng giai đoạn)

1) **Session**
   - Lấy `session_id` từ client, lưu lịch sử chat + `order_state`.
2) **Short memory + Resolve**
   - Gộp ngữ cảnh gần nhất (TTL ~15 phút): anchor SKU, amp, robot/hand, pending parts.
   - Nhận diện follow‑up kiểu “350A”, “500A”, “số lượng 100 cái”.
3) **Intent Detection**
   - Ưu tiên rule‑based (ASK_SELLING_SCOPE, SLOT_FILL_AMP, QUANTITY_FOLLOWUP, bundle).
   - Nếu không rơi vào rule, dùng LLM theo `backend/prompts/intent_detection.txt`.
4) **Routing**
   - CODE_LOOKUP (mã cụ thể), PRODUCT_LOOKUP (quy cách), ACCESSORY_BUNDLE_LOOKUP (đi kèm), LIST, v.v.
5) **Resource Retrieval**
   - Exact lookup cho mã (Tokin / P / D).
   - Bundle retrieval theo từng danh mục yêu cầu (TIP_BODY / INSULATOR / NOZZLE / ORIFICE).
6) **Context Guard**
   - Quyết định: có render sản phẩm không, có form không, có note tay/robot không.
7) **Generation**
   - Bundle/quantity có nhánh xử lý riêng.
   - Còn lại gọi LLM theo `backend/prompts/answer_generation.txt`.
8) **Post‑processing**
   - Gắn ảnh theo Markdown, lọc trùng SKU, chèn câu trung tính (commercial guard).
9) **Persist**
   - Lưu tin nhắn, logs, `order_state` vào `backend/data/sessions.json`.

## Ví dụ luồng hỏi (thực tế)

Câu hỏi: **“Cách điện 004002 dùng chụp khí gì”**

- Nhận diện: có mã + hỏi linh kiện → `ACCESSORY_BUNDLE_LOOKUP`.
- Anchor: SKU `004002` (INSULATOR).
- Required parts: `NOZZLE`.
- Retrieval: lọc nozzle theo amp/system nếu có.
- Trả lời:
  - 1 câu mở đầu ngắn.
  - **Xuất xứ: Tokinarc – Nhật Bản 🇯🇵** (đặt ngay sau mở đầu).
  - Liệt kê chụp khí phù hợp (bullet + ảnh).
  - NOTE tay/robot dạng thông báo (không hỏi bắt buộc).

## Cấu hình model

Các biến môi trường chính:
- `GEMINI_API_KEY` (bắt buộc)
- `GEMINI_MODEL_FLASH` (mặc định `gemini-2.5-flash`)
- `GEMINI_MODEL_PRO` (mặc định `gemini-2.5-pro`)
- `GEMINI_MODEL` (fallback cho cả flash/pro)
- `MAX_IMAGES` (mặc định 4)
- `MAX_ATTEMPTS` (mặc định 3)
- `RESOURCES_PATH` (tuỳ chọn, thay đường dẫn `AgentX.json`)
- `LOG_LEVEL` (INFO/DEBUG)

## Giám sát vận hành & đọc log

Log được in ra terminal theo format:
```
YYYY-MM-DD HH:MM:SS [INFO] autoss.agent: session=... step=...
```

Các dòng quan trọng:
- `session=... question=...` câu hỏi đầu vào
- `intent=... action=...` kết quả intent
- `bundle_query_text=...` / `bundle_filters amp=... system=... anchor_sku=...`
- `bundle_topk group=... results=[...]`
- `step=generation route=...`

Muốn xem log chi tiết hơn:
```
set LOG_LEVEL=DEBUG
uvicorn backend.app:app --reload
```

## Hạn chế & điểm cần khắc phục

- **Chất lượng dữ liệu**: nếu `AgentX.json` thiếu amp/system/robot/hand, kết quả phụ kiện sẽ kém chính xác.
- **Ambiguity**: khi nhiều tuỳ chọn 350A/500A → bot phải hỏi lại để chốt (không tự đoán).
- **Quota Gemini**: free‑tier dễ gặp 429, cần retry/backoff hoặc nâng gói.
- **Encoding**: file prompt hoặc data không UTF‑8 sẽ gây lỗi hiển thị dấu.
- **Demo‑only**: không có auth, session lưu file cục bộ, chỉ giữ 3 session gần nhất.
- **TTL short memory**: sau ~15 phút, follow‑up có thể mất mốc.
