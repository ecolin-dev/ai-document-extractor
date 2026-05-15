"""
Document type classifier.
Uses GPT-4V to identify the type of Mexican vehicle document.
"""
import os
import json
import logging
from openai import OpenAI

log = logging.getLogger("extractor.classifier")

VALID_TYPES = ["ine", "curp", "factura", "pedimento", "repuve", "comprobante_pago", "otro"]


def classify_document(image_b64: str, mime_type: str = "image/jpeg") -> str:
    """
    Classify a document image into a known type.

    Args:
        image_b64: Base64-encoded image
        mime_type: MIME type

    Returns:
        Document type string (ine, curp, factura, etc.)
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

    media = "image/png" if "pdf" in mime_type or "png" in mime_type else "image/jpeg"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": f"""Classify this Mexican document into exactly one type.
Valid types: {', '.join(VALID_TYPES)}

Rules:
- INE/IFE = national voter ID card → "ine"
- CURP document → "curp"
- Vehicle invoice/factura → "factura"
- Import permit (pedimento) → "pedimento"
- REPUVE check → "repuve"
- Government payment receipt → "comprobante_pago"
- Anything else → "otro"

Respond with ONLY the type string, nothing else.""",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What type of document is this?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media};base64,{image_b64}", "detail": "low"},
                    },
                ],
            },
        ],
        max_tokens=20,
        temperature=0.0,
    )

    result = response.choices[0].message.content.strip().lower()

    # Validate
    if result not in VALID_TYPES:
        log.warning(f"Unknown classification: {result}, defaulting to 'otro'")
        return "otro"

    return result
