# Phase 2 — Simple flow (Search & Price)

Phase 2 runs **after** detection (YOLO zones + crops). It turns each crop into a search query, gets web results, and parses price.

---

## Minimal flow

```
image  →  BLIP VQA  →  Build query  →  DuckDuckGo  →  result
```

| Step | What | Where |
|------|------|--------|
| 1 | **Crop** | Per-zone image (from Phase 1) |
| 2 | **BLIP VQA** | Ask: brand?, type?, color? → structured details |
| 3 | **Build query** | details + phase-1 label → one search string (e.g. "... buy price online india") |
| 4 | **DuckDuckGo** | HTML search, get first N results (title + snippet + link) |
| 5 | **Parse price** | From title+snippet: find Rs/₹/INR price; pick highest reasonable; clean digits |
| 6 | **Show** | Overlay: "Rs. &lt;price&gt; &lt;source&gt;" above zone; panel: Source \| Title \| Price |

---

## In code

- **Entry:** User clicks **SEARCH & GET PRICE** → `run_search_and_price()` in `simple_ui.py`
- **Per zone:** `search_product_free(crop_path, product_label)` in `search_product_free.py`
  - `analyze_product(crop_path)` → BLIP VQA → `details`
  - `build_query(details, product_label)` → `query`
  - `free_search(query)` → DuckDuckGo HTML → parse price → return best result (title, price, source, link)
- **UI:** Zones get `search_results`; overlay uses first/reasonable result; panel shows list.

---

## One-line summary

**Phase 2:** Crop → BLIP VQA → query → DuckDuckGo → parse price → show on label + dashboard.

---

## Accuracy / confidence (every run)

The app **does not compute or show a single “accuracy” number** each time.

| Part | What exists |
|------|----------------|
| **Detection (Phase 1)** | Each zone has a **confidence** (0–1), e.g. `0.96`, from the YOLO model. Only boxes above **conf_thresh** are kept: **0.50** (DETECT ALL), **0.35** (RETRY). Confidence is stored in `gui_zones.json` but not shown in the UI. |
| **Phase 2 (BLIP, search, price)** | No accuracy or confidence metric. BLIP and DuckDuckGo results are used as-is; there is no correctness score or evaluation. |

So “accuracy” is **not a fixed value every time** — it varies per detection (per box), and Phase 2 has no accuracy metric. To get a single number you’d need to define it (e.g. “% of zones with confidence &gt; 0.8” or “price match vs ground truth”) and add the logic.
