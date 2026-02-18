"""
DuckDuckGo HTML search + parse price (step 5–6 of pipeline).

Pipeline step: after Build Query → DuckDuckGo (this module) → Parse Price (here).
Returns first ecommerce result with a detectable price (Rs/₹/$); prefer Amazon/Flipkart/Myntra.
"""

import logging
import re
import time
from typing import Dict, Optional

import requests
from bs4 import BeautifulSoup
from urllib.parse import parse_qs, unquote, urljoin, urlparse

logger = logging.getLogger(__name__)


def _resolve_ddg_redirect(href: str) -> str:
    """Extract real destination URL from DuckDuckGo redirect (l/?uddg=...). Returns original if not a DDG redirect."""
    if not href or "duckduckgo.com" not in href.lower():
        return href
    try:
        parsed = urlparse(href)
        if "/l/" in parsed.path or "uddg=" in href:
            qs = parse_qs(parsed.query)
            real = qs.get("uddg", [None])[0]
            if real:
                return unquote(real)
    except Exception:
        pass
    return href


DDG_HTML_URL = "https://html.duckduckgo.com/html/"
# Fallback if html. subdomain is blocked
DDG_HTML_URL_FALLBACK = "https://duckduckgo.com/html/"

# Rupee-only patterns (full price: Rs 89,999 / ₹1,99,999 / INR 89999)
_RUPEE_PATTERNS = [
    re.compile(r"₹\s*[\d,]+(?:\.[\d]{2})?"),
    re.compile(r"Rs\.?\s*[\d,]+(?:\.[\d]{2})?", re.I),
    re.compile(r"INR\s*[\d,]+(?:\.[\d]{2})?", re.I),
    re.compile(r"MRP\s*[:\-]?\s*Rs?\.?\s*[\d,]+(?:\.[\d]{2})?", re.I),
    re.compile(r"(?:price|at|from)\s*[:\-]?\s*Rs?\.?\s*[\d,]+(?:\.[\d]{2})?", re.I),
    re.compile(r"[\d,]+(?:\.[\d]{2})?\s*(?:Rs|₹|INR)\b", re.I),
]

# Legacy: all price patterns (used only when not rupees_only)
_PRICE_PATTERNS = [
    re.compile(r"₹\s*[\d,]+(?:\.[\d]{2})?"),
    re.compile(r"Rs\.?\s*[\d,]+(?:\.[\d]{2})?", re.I),
    re.compile(r"\$\s*[\d,]+(?:\.[\d]{2})?"),
    re.compile(r"USD\s*[\d,]+(?:\.[\d]{2})?", re.I),
    re.compile(r"INR\s*[\d,]+(?:\.[\d]{2})?", re.I),
    re.compile(r"MRP\s*[:\-]?\s*Rs?\.?\s*[\d,]+(?:\.[\d]{2})?", re.I),
    re.compile(r"(?:from|price|at)\s+Rs?\.?\s*[\d,]+(?:\.[\d]{2})?", re.I),
    re.compile(r"[\d,]+(?:\.[\d]{2})?\s*(?:Rs|₹|INR)", re.I),
]


def _extract_domain(url: str) -> str:
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        return host.replace("www.", "")
    except Exception:
        return ""


def _source_from_url(url: str) -> str:
    domain = _extract_domain(url)
    if not domain or "duckduckgo" in domain:
        return "Shop"
    if "amazon" in domain:
        return "Amazon"
    if "flipkart" in domain:
        return "Flipkart"
    if "myntra" in domain:
        return "Myntra"
    if "snapdeal" in domain:
        return "Snapdeal"
    if "croma" in domain:
        return "Croma"
    if "apple" in domain:
        return "Apple"
    if "walmart" in domain:
        return "Walmart"
    if "bestbuy" in domain:
        return "Best Buy"
    if "target" in domain:
        return "Target"
    # Use short domain (e.g. "ebay") as source
    return domain.split(".")[-2].title() if "." in domain else domain.title()


def _find_price(text: str) -> Optional[str]:
    if not text:
        return None
    for pat in _PRICE_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(0).strip()
    return None


def _find_price_rupees(text: str) -> Optional[str]:
    """Extract first price that is in rupees (Rs/₹/INR). Prefer longest match for full amount."""
    if not text:
        return None
    best = None
    best_len = 0
    for pat in _RUPEE_PATTERNS:
        for m in pat.finditer(text):
            s = m.group(0).strip()
            if len(s) > best_len:
                best = s
                best_len = len(s)
    return best


def _price_to_number(price_str: str) -> float:
    """Parse price string (Rs/₹/digits with commas) to a number for comparison. Returns 0 if invalid."""
    if not (price_str or price_str.strip()):
        return 0.0
    s = re.sub(r"[?\uFF1F\uFFFD\u00A0]", "", price_str)  # remove ? and similar
    s = re.sub(r"[^\d.]", "", s)  # keep digits and one decimal dot
    if not s:
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def free_search(query: str, rupees_only: bool = True) -> Dict[str, str]:
    """
    Perform DuckDuckGo HTML search and return first ecommerce result with a price.
    When rupees_only=True (default), only results with Rs/₹/INR price are returned.
    """
    if not query:
        logger.warning("[Search] free_search: empty query, skipping")
        print("[Search] free_search: empty query, skipping")
        return {}

    logger.info("[Search] DuckDuckGo query: %s", query)
    print(f"[Search] DuckDuckGo query: {query!r}")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml",
        "Referer": "https://duckduckgo.com/",
    }
    session = requests.Session()

    def _fetch(url: str):
        try:
            return session.get(url, params={"q": query}, headers=headers, timeout=12)
        except Exception as e:
            logger.exception("[Search] DuckDuckGo request failed: %s", e)
            print(f"[Search] DuckDuckGo request failed: {e}")
            return None

    resp = _fetch(DDG_HTML_URL)
    if resp is None:
        return {}

    logger.info("[Search] DuckDuckGo status=%s len(body)=%s", resp.status_code, len(resp.text))
    print(f"[Search] DuckDuckGo status={resp.status_code} len(body)={len(resp.text)}")

    if resp.status_code not in (200, 202):
        logger.warning("[Search] DuckDuckGo status %s, skipping", resp.status_code)
        print(f"[Search] DuckDuckGo status {resp.status_code}, skipping")
        return {}

    soup = BeautifulSoup(resp.text, "html.parser")
    # Look at up to 20 DuckDuckGo results for pricing
    results = soup.select(".result")[:20]

    # On 202 DDG sometimes returns a "please wait" page with 0 .result; retry once after short delay
    if len(results) == 0 and resp.status_code == 202:
        print("[Search] 202 with 0 results, retrying in 1.5s...")
        time.sleep(1.5)
        resp = _fetch(DDG_HTML_URL)
        if resp and resp.status_code in (200, 202):
            soup = BeautifulSoup(resp.text, "html.parser")
            results = soup.select(".result")[:20]
            print(f"[Search] After retry: status={resp.status_code} .result count={len(results)}")
    if len(results) == 0 and DDG_HTML_URL_FALLBACK != DDG_HTML_URL:
        resp = _fetch(DDG_HTML_URL_FALLBACK)
        if resp and resp.status_code in (200, 202):
            soup = BeautifulSoup(resp.text, "html.parser")
            results = soup.select(".result")[:20]
            print(f"[Search] Fallback URL: .result count={len(results)}")

    logger.info("[Search] DDG .result count: %s", len(results))
    print(f"[Search] DDG .result count: {len(results)}")
    if len(results) == 0:
        print("[Search] WARNING: No .result elements in DDG HTML (page structure may have changed or request blocked)")
        logger.warning("[Search] No .result elements in DDG HTML")

    # Results with a parsed price (preferred)
    with_price = []
    # Known ecommerce without price (fallback so we still show title + source)
    ecommerce_no_price = []
    # For top ecommerce sites, we may do a second pass on the product page
    top_ecom_to_scrape = []
    known_domains = ("amazon", "flipkart", "myntra", "snapdeal", "croma", "reliancedigital", "apple.com")

    for i, r in enumerate(results):
        a = r.select_one("a.result__a")
        if not a:
            continue
        href = a.get("href") or ""
        if not href:
            continue
        href = urljoin(DDG_HTML_URL, href)
        real_url = _resolve_ddg_redirect(href)
        href_lower = real_url.lower()
        title = (a.get_text() or "").strip()
        snippet_el = r.select_one(".result__snippet")
        snippet = (snippet_el.get_text() or "").strip() if snippet_el else ""
        combined = title + " " + snippet
        if rupees_only:
            price = _find_price_rupees(combined)
        else:
            price = _find_price(combined)
        source = _source_from_url(real_url)
        is_known = any(d in href_lower for d in known_domains)
        if i < 5:
            print(f"[Search]   result[{i}] url={real_url[:60]}... price={price!r} known={is_known} snippet={snippet[:60]!r}...")
        # Keep only valid price chars (digits, ., ,, Rs/₹/INR) and normalize formatting
        if price:
            price = re.sub(r"[^\d.,\sRs₹INR]", "", price, flags=re.I).strip() or price
            # Drop trailing commas/periods so we don't log/return 'Rs. 20000,'
            price = price.rstrip(" ,.")
            if price and not re.search(r"\d", price):
                price = ""
        entry = {"title": title, "price": price or "", "source": source, "link": real_url}

        if rupees_only and price:
            # Only add results that have a rupee price
            with_price.append(entry)
        elif not rupees_only:
            if is_known:
                if price:
                    with_price.append(entry)
                else:
                    ecommerce_no_price.append(entry)
            elif price:
                with_price.append(entry)
        elif is_known and not price:
            ecommerce_no_price.append(entry)

        # Track top ecommerce results that didn't expose a price in the snippet
        if is_known and not price:
            top_ecom_to_scrape.append(entry)

    # Second pass: try to fetch price directly from top ecommerce product pages
    # when snippet didn't contain a rupee value.
    if rupees_only and top_ecom_to_scrape:
        for entry in top_ecom_to_scrape:
            url = entry.get("link") or ""
            if not url:
                continue
            try:
                resp = _fetch(url)
                if not resp or resp.status_code != 200 or not resp.text:
                    continue
                page_price = _find_price_rupees(resp.text)
                if not page_price:
                    continue
                # Reuse the same cleaning / normalization logic
                page_price = re.sub(r"[^\d.,\sRs₹INR]", "", page_price, flags=re.I).strip() or page_price
                page_price = page_price.rstrip(" ,.")
                if page_price and re.search(r"\d", page_price):
                    fixed = {**entry, "price": page_price}
                    with_price.append(fixed)
                    logger.info("[Search] Added price from product page: source=%s price=%s url=%s",
                                fixed.get("source"), fixed.get("price"), url)
                    print(f"[Search] Added price from product page: source={fixed.get('source')} price={fixed.get('price')} url={url}")
            except Exception as e:
                logger.warning("[Search] Error scraping price from %s: %s", url, e)
                print(f"[Search] Error scraping price from {url}: {e}")

    # Return result with highest price (and its source); else first ecommerce
    if with_price:
        out = max(with_price, key=lambda e: _price_to_number(e.get("price") or ""))
        logger.info("[Search] Result WITH price (highest): source=%s title=%s price=%s", out.get("source"), (out.get("title") or "")[:50], out.get("price"))
        print(f"[Search] Result WITH price (highest): source={out.get('source')} price={out.get('price')} title={(out.get('title') or '')[:50]!r}")
        return out
    if ecommerce_no_price:
        out = ecommerce_no_price[0]
        logger.info("[Search] Result NO price (ecommerce): source=%s title=%s", out.get("source"), (out.get("title") or "")[:50])
        print(f"[Search] Result NO price: source={out.get('source')} title={(out.get('title') or '')[:50]!r}")
        return out
    logger.warning("[Search] No ecommerce result (with_price=%s, ecommerce_no_price=%s)", len(with_price), len(ecommerce_no_price))
    print(f"[Search] No ecommerce result. with_price={len(with_price)} ecommerce_no_price={len(ecommerce_no_price)}")
    return {}

