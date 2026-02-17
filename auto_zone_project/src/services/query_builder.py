"""Build a single search query from structured product analysis."""
import logging
import re
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def _is_visible(value: str) -> bool:
    if not value:
        return False
    v = value.strip()
    if not v:
        return False
    return v.lower() != "not clearly visible"


def _first_feature_keyword(features: str) -> str:
    """
    Extract a short keyword/phrase from the features string.
    Example: "Two rear cameras, Apple logo centered" -> "dual camera"
    """
    if not _is_visible(features):
        return ""
    # Simple heuristic: take up to first 2 words of first phrase
    first_part = features.split(",")[0]
    words = [w.strip() for w in first_part.split() if w.strip()]
    if not words:
        return ""
    if len(words) >= 2:
        return " ".join(words[:2])
    return words[0]


def _is_misleading_feature_for_category(feat_kw: str, product_label: str) -> bool:
    """Avoid adding feature keyword that steers search to wrong product (e.g. touchpad for cell phone)."""
    if not feat_kw or not product_label:
        return False
    f = feat_kw.lower()
    p = product_label.lower()
    if "phone" in p or "cell" in p or "mobile" in p:
        if "trackpad" in f or "touchpad" in f or "keyboard" in f or "magic" in f:
            return True
    return False


_GENERIC_DETAIL_WORDS = {
    # Too generic alone; often hurts search quality
    "square", "round", "rectangle", "rectangular", "circular",
    "plain", "simple", "smooth",
    "glossy", "matte",
    "pattern", "texture", "design",
    "white", "black", "silver", "gray", "grey",  # color is already added separately
}


def _has_alpha(s: str) -> bool:
    """True if string contains at least one alphabetic character."""
    return bool(re.search(r"[a-zA-Z]", s or ""))


def _short_detail_keyword(value: str) -> str:
    """Extract a short 1–2 word keyword from a detail answer, skipping generic noise."""
    if not _is_visible(value):
        return ""
    s = value.strip().lower()
    s = re.sub(r"[^\w\s-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return ""
    words = [w for w in s.split() if w]
    if not words:
        return ""
    # Prefer first two meaningful words
    stop = {"a", "an", "the", "with", "and", "or", "of", "on", "in", "looks", "like", "visible"}
    words = [w for w in words if w not in stop]
    if not words:
        return ""
    kw = " ".join(words[:2])
    # Never return bare numbers like "3"
    if not _has_alpha(kw):
        return ""
    if kw in _GENERIC_DETAIL_WORDS or words[0] in _GENERIC_DETAIL_WORDS:
        return ""
    return kw


def _parts_layout_keyword(parts_layout: str) -> str:
    """
    Extract a strong identifying keyword from a parts/layout answer.
    Examples:
      - "two lenses arranged vertically" -> "dual lenses vertical"
      - "3 buttons on the side" -> "triple buttons"
    """
    if not _is_visible(parts_layout):
        return ""
    s = parts_layout.strip().lower()
    s = re.sub(r"[^\w\s-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # Count normalization
    count = ""
    if re.search(r"\b(2|two|dual|double)\b", s):
        count = "dual"
    elif re.search(r"\b(3|three|triple)\b", s):
        count = "triple"
    elif re.search(r"\b(4|four|quad)\b", s):
        count = "quad"

    # Layout (optional)
    layout = ""
    if re.search(r"\b(vertical|vertically)\b", s):
        layout = "vertical"
    elif re.search(r"\b(horizontal|horizontally)\b", s):
        layout = "horizontal"

    noun = ""
    m = re.search(r"\b(?:2|two|dual|double|3|three|triple|4|four|quad)\b\s+([a-z][a-z0-9_-]{2,})", s)
    if m:
        noun = m.group(1)

    if count and noun:
        kw = f"{count} {noun} {layout}".strip()
        return kw if _has_alpha(kw) else ""

    # Fallback to the existing visual-details logic (count/layout + short phrase)
    kw = _visual_detail_keyword(s)
    return kw if _has_alpha(kw) else ""


def _visual_detail_keyword(visual_details: str) -> str:
    """
    Turn BLIP visual-details answer into a short keyword/phrase usable in search.

    Examples:
      - "two rear cameras arranged vertically" -> "dual camera vertical"
      - "three buttons on the side" -> "three buttons"
      - "striped pattern" -> "striped pattern"
    """
    if not _is_visible(visual_details):
        return ""
    s = visual_details.strip().lower()

    # Normalize whitespace/punctuation
    s = re.sub(r"[\[\](){}]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # Count
    count = ""
    if re.search(r"\b(2|two|dual|double)\b", s):
        count = "dual"
    elif re.search(r"\b(3|three|triple)\b", s):
        count = "triple"
    elif re.search(r"\b(4|four|quad)\b", s):
        count = "quad"

    # Layout (optional)
    layout = ""
    if re.search(r"\b(vertical|vertically)\b", s):
        layout = "vertical"
    elif re.search(r"\b(horizontal|horizontally)\b", s):
        layout = "horizontal"

    # Object noun after a count (camera/buttons/stripes/etc.)
    noun = ""
    m = re.search(r"\b(?:2|two|dual|double|3|three|triple|4|four|quad)\b\s+([a-z][a-z0-9_-]{2,})", s)
    if m:
        noun = m.group(1)

    if count and noun:
        if layout:
            return f"{count} {noun} {layout}"
        return f"{count} {noun}"

    # Fallback: return first 2–3 words if they look meaningful
    words = [w for w in re.split(r"\s+", s) if w]
    if not words:
        return ""
    # Drop very common filler words
    stop = {"a", "an", "the", "with", "and", "or", "of", "on", "in", "visible", "looks", "like"}
    words = [w for w in words if w not in stop]
    if not words:
        return ""
    return " ".join(words[:3])


def build_query(details: Dict[str, str], product_label: Optional[str] = None) -> str:
    """
    Build a single search query string.

    Logic:
        - Include brand if visible
        - Always include product_type (from BLIP)
        - Include phase-1 product_label when provided (e.g. "cell phone") so search stays in right category
        - Include color if visible
        - Include first feature keyword only if not misleading for category
        - Append 'buy price online'
    """
    parts = []

    brand = details.get("brand", "")
    product_type = details.get("product_type", "")
    color = details.get("color", "")
    features = details.get("features", "")
    visual_details = details.get("visual_details", "")
    shape = details.get("shape", "")
    material = details.get("material", "")
    pattern_texture = details.get("pattern_texture", "")
    parts_layout = details.get("parts_layout", "")
    distinctive_marking = details.get("distinctive_marking", "")

    if _is_visible(brand):
        parts.append(brand)
    if _is_visible(product_type):
        parts.append(product_type)
    else:
        parts.append("product")
    # Phase-1 label (e.g. "cell phone") anchors search to detected category so we don't get trackpad/keyboard results
    if product_label and product_label.strip():
        pl = product_label.strip()
        if pl.lower() not in {p.lower() for p in parts}:
            parts.append(pl)
    if _is_visible(color):
        parts.append(color)

    feat_kw = _first_feature_keyword(features)
    if feat_kw and not _is_misleading_feature_for_category(feat_kw, product_label or ""):
        parts.append(feat_kw)

    # Extra details (generic): keep it small to avoid noisy queries
    extra = []
    # Prefer parts/layout + distinctive marking first (highest identifying power)
    for v, fn in (
        (parts_layout, _parts_layout_keyword),
        (distinctive_marking, _short_detail_keyword),
        (material, _short_detail_keyword),
        (pattern_texture, _short_detail_keyword),
        (shape, _short_detail_keyword),
    ):
        kw = fn(v)
        if kw and kw.lower() not in {p.lower() for p in parts} and kw.lower() not in {e.lower() for e in extra}:
            extra.append(kw)
        if len(extra) >= 2:
            break

    # Fallback detail: visual_details (only if we still have room)
    if len(extra) < 2:
        vd_kw = _visual_detail_keyword(visual_details)
        if vd_kw and not _is_misleading_feature_for_category(vd_kw, product_label or ""):
            if vd_kw.lower() not in {p.lower() for p in parts} and vd_kw.lower() not in {e.lower() for e in extra}:
                extra.append(vd_kw)

    parts.extend(extra)

    parts.append("buy price online india")
    query = " ".join(parts)
    logger.info("[Query] build_query: product_label=%s parts=%s -> %s", product_label, parts, query)
    print(f"[Query] build_query: product_label={product_label!r} parts={parts} -> {query!r}")
    return query

