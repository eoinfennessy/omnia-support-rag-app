"""Text chunking and preprocessing for different document types."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from omnia_support_models import DocumentType, ProductCategory

if TYPE_CHECKING:
    from omnia_support_models import Document

_SEPARATORS = ("\n\n", ". ", ".\n", "? ", "! ")
_TICKET_CUSTOMER_RE = re.compile(r"^Customer:.*\n", re.MULTILINE)
_TICKET_ORDER_RE = re.compile(r"^Order:\s*ORD-[\w-]+\s*—\s*", re.MULTILINE)
_POLICY_HEADER_RE = re.compile(
    r"^OMNIA RETAIL LTD\s*[—-]\s*.+\n(Effective Date:.+\n)?",
    re.MULTILINE,
)


def _find_break_point(text: str, start: int, end: int, search_back: int = 400) -> int:
    """Find the best position to split text, preferring sentence boundaries."""
    region_start = max(start, end - search_back)
    region = text[region_start:end]

    for sep in _SEPARATORS:
        idx = region.rfind(sep)
        if idx != -1:
            return region_start + idx + len(sep)

    idx = text.rfind(" ", start, end)
    return idx + 1 if idx != -1 else end


def chunk_text(
    text: str,
    max_chars: int = 2000,
    overlap: int = 200,
) -> list[str]:
    """Split text into chunks at sentence/paragraph boundaries."""
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            end = _find_break_point(text, start, end)
        chunks.append(text[start:end])
        start = max(start + 1, end - overlap)

    return chunks


def _preprocess_ticket(content: str) -> str:
    """Strip noise fields from support ticket content.

    Removes the Customer line and order number prefix, keeping the product description.
    """
    content = _TICKET_CUSTOMER_RE.sub("", content)
    content = _TICKET_ORDER_RE.sub("Product: ", content)
    return content.strip()


def _chunk_policy(content: str) -> list[str]:
    """Strip the top-level heading and split at section headings."""
    content = _POLICY_HEADER_RE.sub("", content)
    sections = re.split(r"\n\n(?=[A-Z\d])", content)
    return [s.strip() for s in sections if s.strip()]


def chunk_document(doc: Document) -> list[str]:
    """Chunk a document using a type-appropriate strategy."""
    if doc.type == DocumentType.FAQ:
        return [doc.content]
    if doc.type == DocumentType.POLICY:
        return _chunk_policy(doc.content)
    content = _preprocess_ticket(doc.content)
    return chunk_text(content)


def build_embed_text(doc: Document, chunk: str) -> str:
    """Prepend metadata to a chunk for richer embeddings."""
    lines = [f"Type: {doc.type.value}", f"Category: {doc.category}"]
    if doc.product_category not in (ProductCategory.ALL, ProductCategory.NONE):
        lines.append(f"Product: {doc.product_category.value}")
    return "\n".join(lines) + "\n" + chunk.strip()
