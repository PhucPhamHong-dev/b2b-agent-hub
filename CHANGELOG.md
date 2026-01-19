# Changelog

All notable changes to this project are documented in this file.

## Unreleased
- Added short-memory slots and follow-up resolution to keep anchors across turns.
- Added pending action handling so short replies (e.g., "muon", "ok", "350A") route correctly.
- Improved bundle retrieval to respect required parts and avoid unrelated categories.
- Added quantity-followup prompt handling with form enforcement when needed.
- Added stock-status line support driven by numeric "Don vi" values.
- Added richer retrieval logging for bundle queries (query text, filters, top-k).
- Added product card/image guards to reduce duplicate SKU output and missing images.
- Added two-file knowledge memory (core/delta) with retrieval and gated append updates.
