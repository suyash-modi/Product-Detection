"""
Backward-compat wrapper around the new CPU-only search pipeline.

Existing UI may still import search_product_free from this module, so we
delegate to src.services.search_product_free.search_product_free and
adapt the return type if needed.
"""
from typing import List, Dict

try:
    from .search_product_free import search_product_free as _new_search_product_free
except ImportError:
    from src.services.search_product_free import search_product_free as _new_search_product_free


def search_product_free(crop_path, product_label=None) -> List[Dict]:
    """
    UI adapter: keep UI expecting a list of results.

    Under the hood, `src.services.search_product_free.search_product_free()` returns a single dict
    (analysis + best search result). This wrapper converts that into a list with one UI-friendly item.
    """
    if not crop_path:
        return []
    result = _new_search_product_free(crop_path, product_label) or {}
    if not result:
        return []
    # Display title: prefer title from search source (e.g. product name from Amazon/Flipkart)
    source_title = (result.get("title") or "").strip()
    if source_title:
        title = source_title
    else:
        # Fallback: use analyzer (brand, product_type) or zone label
        _nv = "not clearly visible"
        title_parts = []
        for key in ("brand", "product_type"):
            val = (result.get(key) or "").strip()
            if val and val.lower() != _nv:
                title_parts.append(val)
        title = " ".join(title_parts).strip() or (product_label or "product")
    item = {
        "title": title,
        "price": result.get("price"),
        "source": result.get("source"),
        "link": result.get("link"),
        "analysis": {
            "brand": result.get("brand"),
            "product_type": result.get("product_type"),
            "color": result.get("color"),
            "features": result.get("features"),
        },
    }
    return [item]

