# Sales Agent (Tokinarc) – Demo

Hệ thống demo Sales Assistant B2B cho phụ kiện MIG/MAG Tokinarc, chạy FastAPI + Gemini API + pipeline theo ADK, có UI đơn giản và thinking logs hiển thị.

## 1) Cài đặt & chạy
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

Tạo `backend/.env`:
```
GEMINI_API_KEY=YOUR_KEY_HERE
GEMINI_MODEL_FLASH=gemini-2.5-flash
GEMINI_MODEL_PRO=gemini-2.5-pro
MAX_ATTEMPTS=3
MAX_IMAGES=4
# Self-learning
KNOWLEDGE_ENABLED=1
KNOWLEDGE_TOPK=6
KNOWLEDGE_MAX_NEW_LINES=5
```

Chạy API:
```powershell
uvicorn backend.app:app --reload
```
Mở UI: http://127.0.0.1:8000

## 2) Nguồn dữ liệu
- Dữ liệu sản phẩm: `resources/AgentX.json` (đọc mỗi request, không bịa thêm).
- Short memory phiên: `backend/data/sessions.json` (TTL ~15 phút, giữ anchor SKU/amp/tay-robot/pending parts).
- Knowledge dài hạn: `knowledge/knowledge_core.md` (ổn định) + `knowledge/knowledge_delta.md` (append-only do LLM học thêm).

## 3) Nguyên lý vận hành (các bước chính)
1) **Session**: nhận `session_id`, nạp `order_state`, history.
2) **Short memory + Resolve**: gộp ngữ cảnh gần nhất (anchor SKU, amp, tay/robot, pending parts, constraints).
3) **Intent Detection**: rule trước (ASK_SELLING_SCOPE, SLOT_FILL_AMP, QUANTITY_FOLLOWUP, bundle…) rồi LLM (`backend/prompts/intent_detection.txt`) nếu cần.
4) **Routing**: CODE_LOOKUP, PRODUCT_LOOKUP, ACCESSORY_BUNDLE_LOOKUP, LIST, v.v.
5) **Resource Retrieval**: exact lookup (Tokin/P/D), bundle theo danh mục yêu cầu (TIP_BODY/INSULATOR/NOZZLE/ORIFICE), lọc amp/system/type.
6) **Knowledge Retrieve**: lấy top K chunk từ core+delta (`backend/knowledge/knowledge_store.py` → `knowledge/md_index.json`) đưa vào prompt.
7) **Context Guard**: quyết định render sản phẩm, form, note tay/robot.
8) **Generation**: nhánh rule (bundle/quantity) hoặc LLM (`backend/prompts/answer_generation.txt`).
9) **Post-processing**: gắn ảnh Markdown, lọc trùng SKU, chèn câu trung tính thương mại.
10) **Persist + Self-learning**: lưu session; LLM đề xuất tri thức mới → gate → append delta (`backend/knowledge/knowledge_updater.py`).

## 4) Self-learning hai tầng
- **retrieve**: chunk core + delta, score từ khóa, ưu tiên delta khi tie, chỉ lấy topK (KNOWLEDGE_TOPK) để tiết kiệm token, chèn vào prompt dưới block `KNOWLEDGE CONTEXT`.
- **update**: sau khi trả lời, gọi Gemini với `backend/prompts/knowledge_extractor.txt` để đề xuất ≤ KNOWLEDGE_MAX_NEW_LINES; bộ lọc chặn injection, chỉ nhận TAG (QA/SYN/RULE/TEMPLATE), tiếng Việt, SKU/spec phải có trong AgentX; dedupe trước khi append vào `knowledge_delta.md` (ghi atomically).
- **Lợi ích**: giữ rule/synonym/template/QA hay dùng mà không nhét toàn bộ vào prompt; dễ kiểm soát vì chỉ cần đọc core+delta.

### Ví dụ luồng hỏi + tự học
1) User: “Cách điện 004002 dùng chụp khí gì?”
2) Intent: ACCESSORY_BUNDLE_LOOKUP, anchor=004002 (INSULATOR), required_parts=NOZZLE.
3) Retrieval: lọc nozzle theo amp/system (nếu có) từ AgentX.
4) Knowledge retrieve: lấy topK rule/template (note tay/robot, mở đầu/closing) từ core+delta để hỗ trợ LLM.
5) Answer: chào ngắn → Xuất xứ → bullet nozzle kèm ảnh → NOTE tay/robot dạng thông báo.
6) Knowledge update: LLM đề xuất dòng mới (ví dụ synonym “chụp khí” = “nozzle”, rule hỏi amp khi nhiều tùy chọn) → gate → append `knowledge_delta.md`. Lần sau retrieval sẽ ưu tiên delta nếu liên quan.

## 5) Ví dụ luồng hỏi (thực tế)
Câu: **“Cách điện 004002 dùng chụp khí gì”**
- Nhận diện: có mã + hỏi linh kiện → `ACCESSORY_BUNDLE_LOOKUP`.
- Anchor: SKU `004002` (INSULATOR).
- Required parts: `NOZZLE`.
- Retrieval: lọc nozzle theo amp/system nếu có.
- Trả lời: chào ngắn; **Xuất xứ: Tokinarc – Nhật Bản 🇯🇵** (đặt ngay sau mở đầu); liệt kê chụp khí phù hợp (bullet + ảnh); NOTE tay/robot dạng thông báo (không hỏi bắt buộc).

## 6) Cấu hình model
- `GEMINI_MODEL_FLASH` (mặc định `gemini-2.5-flash`), `GEMINI_MODEL_PRO` (mặc định `gemini-2.5-pro`), `GEMINI_MODEL` (fallback).
- `MAX_IMAGES` (mặc định 4), `MAX_ATTEMPTS` (mặc định 3).
- `RESOURCES_PATH` (tùy chọn), `LOG_LEVEL` (INFO/DEBUG).

## 7) Giám sát vận hành & đọc log
Log format:
```
YYYY-MM-DD HH:MM:SS [INFO] autoss.agent: session=... step=...
```
Các dòng quan trọng: `question=...`, `intent=... action=...`, `bundle_query_text=...`, `bundle_filters amp=... system=... anchor_sku=...`, `bundle_topk group=... results=[...]`, `knowledge_retrieve topk=...`, `knowledge_update appended_lines=...`, `step=generation route=...`.
Xem chi tiết hơn: `set LOG_LEVEL=DEBUG && uvicorn backend.app:app --reload`

## 8) Hạn chế & điểm cần khắc phục
- Dữ liệu AgentX thiếu amp/system/robot/hand → bundle có thể không khớp.
- Ambiguity: nhiều lựa chọn 350A/500A → phải hỏi lại, không tự đoán.
- Ảnh trùng URL: logic chèn ảnh tránh lặp URL, nên chuẩn bị ảnh khác nhau cho từng SKU nếu muốn hiển thị đầy đủ.
- Quota Gemini: free-tier dễ 429; cần retry/backoff hoặc nâng gói.
- Demo-only: không auth, session lưu file cục bộ, chỉ giữ vài session gần nhất.
- TTL short memory: sau ~15 phút, follow-up có thể mất mốc.
