"""
Product search pipeline: Crop → BLIP VQA → Build Query → DuckDuckGo → Parse Price.

Flow (matches app):
  1. OpenVINO (YOLO)  → detect product zones
  2. Crop             → save per-zone crop image (crop_path)
  3. BLIP VQA         → ask 3–4 targeted questions (brand, type, color, features)
  4. Build Query      → build search string from VQA answers (+ YOLO label fallback)
  5. DuckDuckGo       → HTML search with that query
  6. Parse Price      → first result with price; return title, price, source, link
"""
import logging
from typing import Dict, Optional

from .product_analyzer import analyze_product
from .query_builder import build_query
from .free_search import free_search

logger = logging.getLogger(__name__)
_NOT_VISIBLE = "not clearly visible"
_NOT_CLEAR_TITLE = "Not a clear image"


def search_product_free(crop_path: str, product_label: Optional[str] = None) -> Dict[str, str]:
    """
    Run pipeline: BLIP VQA on crop → build query → DuckDuckGo → parse price.
    When analyzer is off or returns "Not clearly visible", query uses zone label (YOLO).
    """
    logger.info("[Pipeline] search_product_free crop_path=%s product_label=%s", crop_path, product_label)
    print(f"[Pipeline] search_product_free crop_path={crop_path!r} product_label={product_label!r}")

    details = analyze_product(crop_path)
    logger.info("[Pipeline] BLIP VQA details: brand=%s product_type=%s color=%s", details.get("brand"), details.get("product_type"), details.get("color"))
    print(f"[Pipeline] BLIP VQA: brand={details.get('brand')!r} product_type={details.get('product_type')!r} color={details.get('color')!r}")

    # Brand clarity gate (single BLIP pass):
    # If BLIP isn't 100% sure about brand, skip Phase-2 search/price entirely.
    brand = (details.get("brand") or "").strip().lower()
    if (not brand) or (brand == _NOT_VISIBLE):
        logger.info("[Pipeline] Brand not clear (brand=%r). Skipping search/price.", details.get("brand"))
        print(f"[Pipeline] Brand not clear (brand={details.get('brand')!r}). Skipping search/price.")
        return {
            "brand": "Not clearly visible",
            "product_type": details.get("product_type") or "Not clearly visible",
            "color": details.get("color") or "Not clearly visible",
            "features": details.get("features") or "Not clearly visible",
            "title": _NOT_CLEAR_TITLE,
            "price": "",
            "source": "",
            "link": "",
        }

    # Use zone label for query only when analyzer gave nothing useful (model off or failed)
    pt = (details.get("product_type") or "").strip().lower()
    if not pt or pt == _NOT_VISIBLE:
        if product_label and product_label.strip():
            details = {**details, "product_type": product_label.strip()}
            logger.info("[Pipeline] Using YOLO label for product_type: %s", product_label.strip())
            print(f"[Pipeline] Using YOLO label for product_type: {product_label.strip()!r}")

    query = build_query(details, product_label=product_label)
    logger.info("[Pipeline] Built query: %s", query)
    print(f"[Pipeline] Built query: {query!r}")

    result = free_search(query)
    has_price = bool((result.get("price") or "").strip())
    logger.info("[Pipeline] free_search result: title=%s source=%s price=%s has_price=%s", result.get("title"), result.get("source"), result.get("price"), has_price)
    print(f"[Pipeline] free_search result: title={result.get('title')!r} source={result.get('source')!r} price={result.get('price')!r} has_price={has_price}")

    return {
        "brand": details.get("brand", ""),
        "product_type": details.get("product_type", ""),
        "color": details.get("color", ""),
        "features": details.get("features", ""),
        # From search result (source) – preferred for display
        "title": result.get("title"),
        "price": result.get("price"),
        "source": result.get("source"),
        "link": result.get("link"),
    }

