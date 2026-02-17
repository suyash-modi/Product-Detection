"""
CPU-only product analysis using BLIP VQA (Salesforce/blip-vqa-base).

Structured extraction by asking specific questions instead of captioning:
  - What brand is this?
  - What color is this product?
  - What product type is this?
  - (optional) features and visible text

Lightweight and more reliable than full captioning.

Set USE_PRODUCT_ANALYZER=0 in .env to skip and use YOLO-label search only.
"""
import logging
import os
import re
from typing import Dict

from PIL import Image

import torch
from transformers import BlipForQuestionAnswering, BlipProcessor

logger = logging.getLogger(__name__)

# Load .env from workspace or auto_zone_project
try:
    from dotenv import load_dotenv
    for _base in (
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
    ):
        _env = os.path.join(_base, ".env")
        if os.path.isfile(_env):
            load_dotenv(_env)
            break
except Exception:
    pass

_USE_ANALYZER = os.environ.get("USE_PRODUCT_ANALYZER", "1").strip().lower() not in ("0", "false", "no")

_MODEL_NAME = "Salesforce/blip-vqa-base"
_processor = None
_model = None

# VQA questions for structured extraction (better than captioning)
_VQA_QUESTIONS: Dict[str, str] = {
    "brand": "What brand is this? Only answer if you are 100% sure; otherwise reply 'Not clearly visible'.",
    "brand_logo": "What brand logo is visible on this product? Only answer if you are 100% sure; otherwise reply 'Not clearly visible'.",
    "product_type": "What product type is this?",
    "color": "What color is this product?",
    "features": "What key features are visible on this product?",
    "visible_text": "What text or writing is visible on this product?",
    # Deeper, generic details (works for any product/category)
    "visual_details": "Describe distinctive visible details (shape, pattern, parts count, arrangement) in a few words.",
    "shape": "What is the overall shape (round, square, rectangular, etc.)?",
    "material": "What material does it look like (plastic, metal, glass, paper, fabric)?",
    "pattern_texture": "What pattern/texture is visible (striped, dotted, plain, glossy, matte)?",
    "parts_layout": "If there are multiple similar parts (e.g. buttons, holes, stripes, lenses), how many and how are they arranged (vertical/horizontal)?",
    "distinctive_marking": "What distinctive marking or symbol is visible (logo shape, icon, label) in a few words?",
}


def _load_model():
    """Lazy-load BLIP VQA once, CPU-only (skipped if USE_PRODUCT_ANALYZER=0)."""
    global _processor, _model
    if not _USE_ANALYZER:
        return
    if _model is not None and _processor is not None:
        return

    logger.info("[ProductAnalyzer] Loading BLIP VQA on CPU: %s", _MODEL_NAME)
    try:
        _processor = BlipProcessor.from_pretrained(_MODEL_NAME)
        _model = BlipForQuestionAnswering.from_pretrained(_MODEL_NAME)
        _model = _model.to("cpu")
        _model.eval()
        logger.info("[ProductAnalyzer] BLIP VQA loaded successfully on CPU")
    except Exception as e:
        logger.error("[ProductAnalyzer] Failed to load BLIP VQA: %s", e, exc_info=True)
        _processor = None
        _model = None


def _default_result() -> Dict[str, str]:
    return {
        "brand": "Not clearly visible",
        "brand_logo": "Not clearly visible",
        "product_type": "Not clearly visible",
        "color": "Not clearly visible",
        "features": "Not clearly visible",
        "visible_text": "Not clearly visible",
        "visual_details": "Not clearly visible",
        "shape": "Not clearly visible",
        "material": "Not clearly visible",
        "pattern_texture": "Not clearly visible",
        "parts_layout": "Not clearly visible",
        "distinctive_marking": "Not clearly visible",
    }


def _norm_brand(s: str) -> str:
    """Normalize brand for strict comparisons (very conservative)."""
    if not s:
        return ""
    s = s.strip().lower()
    if not s or s == "not clearly visible":
        return ""
    # Remove punctuation/spaces so 'Coca Cola' and 'coca-cola' compare equal
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def _clean_brand(s: str) -> str:
    """Clean brand-like answers into a display-friendly form; return '' if unclear."""
    if not s or s.strip().lower() == "not clearly visible":
        return ""
    out = s.strip()
    # Remove leading articles
    for p in ("a ", "an ", "the "):
        if out.lower().startswith(p):
            out = out[len(p):].strip()
            break
    # Remove common trailing words
    for suf in (" logo", " brand", " company", " product"):
        if out.lower().endswith(suf):
            out = out[: -len(suf)].strip()
    return out


def _normalize_answer(raw: str) -> str:
    """Map empty or vague answers to 'Not clearly visible'."""
    if not raw or not raw.strip():
        return "Not clearly visible"
    raw = raw.strip()
    lower = raw.lower()
    if lower in ("unknown", "n/a", "na", "none", "no", "nothing", "unclear", "cannot see"):
        return "Not clearly visible"
    # Treat uncertainty as not visible/clear
    if any(k in lower for k in ("not sure", "unsure", "maybe", "probably", "i think", "seems like", "might be")):
        return "Not clearly visible"
    if lower.startswith("i don't") or lower.startswith("i cannot") or "not visible" in lower:
        return "Not clearly visible"
    return raw


def _ask_vqa(image: Image.Image, question: str) -> str:
    """Run one VQA question on the image; returns decoded answer string."""
    inputs = _processor(images=image, text=question, return_tensors="pt")
    inputs = {k: v.to(_model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = _model.generate(**inputs, max_length=50)
    answer = _processor.decode(out[0], skip_special_tokens=True)
    return _normalize_answer(answer)


def analyze_product(image_path: str) -> Dict[str, str]:
    """
    Analyze a cropped product image using BLIP VQA and return structured details.

    Returns dict with keys: brand, brand_logo, product_type, color, features, visible_text, visual_details.
    """
    if not _USE_ANALYZER:
        return _default_result()
    _load_model()
    if _model is None or _processor is None:
        return _default_result()

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        logger.error("[ProductAnalyzer] Failed to open image %s: %s", image_path, e)
        return _default_result()

    # BLIP expects 384x384 by default; resize for consistency (processor may resize internally)
    if max(img.size) != 384:
        img = img.resize((384, 384), Image.BICUBIC)

    result = _default_result()
    try:
        for key, question in _VQA_QUESTIONS.items():
            answer = _ask_vqa(img, question)
            result[key] = answer
    except Exception as e:
        logger.warning("[ProductAnalyzer] VQA error: %s", e)
        return _default_result()

    # Brand must be extremely strict:
    # accept brand ONLY if brand-question and logo-question agree after normalization.
    b1 = _clean_brand(result.get("brand", ""))
    b2 = _clean_brand(result.get("brand_logo", ""))
    if _norm_brand(b1) and _norm_brand(b1) == _norm_brand(b2):
        # Prefer the logo answer casing if available
        result["brand"] = b2 or b1
    else:
        result["brand"] = "Not clearly visible"

    return result
