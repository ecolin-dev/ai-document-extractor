"""
PDF processing utilities.
Converts PDFs to images for GPT-4V analysis.
Handles large files with compression.
"""
import io
import base64
import logging
from typing import List, Optional

log = logging.getLogger("extractor.pdf")


def pdf_to_images(pdf_bytes: bytes, max_pages: int = 5, max_size_mb: int = 20) -> List[str]:
    """
    Convert PDF pages to base64-encoded PNG images.

    Args:
        pdf_bytes: Raw PDF file bytes
        max_pages: Maximum pages to process
        max_size_mb: Max file size before compression

    Returns:
        List of base64-encoded PNG strings
    """
    import pdfplumber
    from PIL import Image

    size_mb = len(pdf_bytes) / (1024 * 1024)

    # Compress large PDFs
    if size_mb > max_size_mb:
        pdf_bytes = _compress_pdf(pdf_bytes)
        log.info(f"Compressed PDF: {size_mb:.1f}MB → {len(pdf_bytes) / (1024*1024):.1f}MB")

    images = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for i, page in enumerate(pdf.pages[:max_pages]):
                img = page.to_image(resolution=200)
                buf = io.BytesIO()
                img.original.save(buf, format="PNG", optimize=True)
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                images.append(b64)
                log.info(f"Converted page {i+1}/{min(len(pdf.pages), max_pages)}")
    except Exception as e:
        log.error(f"PDF conversion error: {e}")

    return images


def _compress_pdf(pdf_bytes: bytes) -> bytes:
    """
    Compress a large PDF by reducing image resolution.
    Falls back to original if compression fails.
    """
    try:
        import pdfplumber
        from PIL import Image

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            images = []
            for page in pdf.pages:
                img = page.to_image(resolution=150)
                images.append(img.original)

        if not images:
            return pdf_bytes

        buf = io.BytesIO()
        first = images[0]
        if len(images) > 1:
            first.save(buf, format="PDF", save_all=True, append_images=images[1:], optimize=True)
        else:
            first.save(buf, format="PDF", optimize=True)

        compressed = buf.getvalue()
        log.info(f"PDF compressed: {len(pdf_bytes)} → {len(compressed)} bytes")
        return compressed

    except Exception as e:
        log.warning(f"Compression failed, using original: {e}")
        return pdf_bytes


def extract_text(pdf_bytes: bytes) -> str:
    """Extract raw text from PDF using pdfplumber."""
    import pdfplumber

    text_parts = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                text_parts.append(text)
    except Exception as e:
        log.error(f"Text extraction error: {e}")

    return "\n\n".join(text_parts)
