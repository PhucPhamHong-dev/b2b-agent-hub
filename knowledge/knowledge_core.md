# Knowledge Core

## Purpose
- [2026-01-16][RULE][high] Use the catalog as the only source of product facts (SKU/specs/images).
- [2026-01-16][RULE][high] For technical intents, avoid internal handoff phrases unless a contact form is requested.

## Rules
- [2026-01-16][RULE][high] If user asks to list 2-4 items but only 1 SKU matches constraints, state only that 1 SKU and ask to broaden constraints (amp/system/size/robot). Never invent extra SKUs.
- [2026-01-16][RULE][medium] Knowledge coordination: core is stable, delta is append-only; retrieval should dedupe across core+delta and prefer delta when relevance is tied.
- [2026-01-16][RULE][high] If user asks for generic robot accessories without anchor/amp/system, do NOT claim default hand; ask for amp (350A/500A) and system (N/D) or a reference SKU/torch model before listing. Skip origin/hand notes when no product is shown.
- [2026-01-16][RULE][high] Default typing: LIST/technical answers render for HAND unless user explicitly says robot; when is_robot is true, do not append default hand note.
- [2026-01-16][RULE][high] Origin line template: “Xuất xứ: Tokinarc – Nhật Bản 🇯🇵” placed immediately after greeting/opening.
- [2026-01-16][RULE][high] Technical closing lines are short CTA only (no contact unless show_form), e.g. offer to list synced components, ask amp/system if missing.
- [2026-01-16][RULE][high] Handoff/CS phrases are forbidden unless form is shown (e.g., “ghi nhận nhu cầu”, “chuyển bộ phận”, “để kho kiểm tra”).

## Synonyms
- [2026-01-16][SYN][medium] "than giu bec" => TIP_BODY
- [2026-01-16][SYN][medium] "cach dien" => INSULATOR
- [2026-01-16][SYN][medium] "chup khi" => NOZZLE
- [2026-01-16][SYN][medium] "su phan phoi khi" => ORIFICE

## Templates
- [2026-01-16][TEMPLATE][high] NOTE tay/robot (inform, not ask): “Hiện em đang đối chiếu theo súng hàn tay MIG {amp?}; nếu dùng Robot Anh/Chị báo em để em đối chiếu lại mã robot ạ.” Avoid when is_robot=true.
- [2026-01-16][TEMPLATE][high] Closing technical (no contact unless form): pick one CTA: offer more accessories in same system, ask amp/system if missing, offer images/links.
- [2026-01-16][TEMPLATE][high] Default price guard: commercial answers end with neutral sentence, no commitments on stock/price unless form flow.
