"""
Scrape text content from a URL.

Uses httpx for fetching and BeautifulSoup for HTML cleaning.
Attempts to find the main article/content area; falls back to full body text.
"""
from __future__ import annotations

import re


# Tags that never contain useful text content
_REMOVE_TAGS = [
    "script", "style", "nav", "footer", "header",
    "aside", "form", "noscript", "iframe", "svg",
    "button", "input", "select", "textarea",
]

# CSS class/id hints that indicate the main content area
_CONTENT_HINTS = [
    "article", "main", "content", "post", "entry",
    "body-text", "article-body", "post-body",
]


def scrape_url(url: str, timeout: int = 30) -> tuple[str, str]:
    """
    Fetch and clean text from a URL.

    Returns:
        (title, body_text) — title may be empty string if not found.

    Raises httpx.HTTPStatusError on non-2xx responses.
    """
    import httpx
    from bs4 import BeautifulSoup

    resp = httpx.get(
        url,
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 (compatible; LLMPlatform/1.0; +research)"},
    )
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")

    # Page title
    title = soup.title.string.strip() if soup.title and soup.title.string else ""

    # Remove noise elements
    for tag in soup(_REMOVE_TAGS):
        tag.decompose()

    # Try to find main content container
    content_el = None
    for hint in _CONTENT_HINTS:
        content_el = (
            soup.find("main")
            or soup.find("article")
            or soup.find(id=re.compile(hint, re.I))
            or soup.find(class_=re.compile(hint, re.I))
        )
        if content_el:
            break

    target = content_el or soup.body or soup

    # Extract clean text
    raw = target.get_text(separator="\n", strip=True)

    # Collapse excessive blank lines
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    # Remove very short lines (nav remnants, button labels, etc.)
    lines = [l for l in lines if len(l) > 20]
    text = "\n".join(lines)

    if not text.strip():
        raise ValueError(f"No extractable text found at {url}")

    return title, text
