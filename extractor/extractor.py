"""
Document extraction using GPT-4 Vision.
Sends document images to GPT-4V with type-specific prompts.
"""
import os
import json
import logging
from pathlib import Path
from openai import OpenAI

log = logging.getLogger("extractor.extract")

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

# Document type → prompt file mapping
PROMPT_MAP = {
    "ine": "ine.txt",
    "curp": "curp.txt",
    "factura": "factura.txt",
    "pedimento": "pedimento.txt",
    "repuve": "repuve.txt",
    "comprobante_pago": "comprobante.txt",
}

# Base extraction schema all documents share
BASE_SCHEMA = {
    "vehiculo": {
        "marca": None, "linea": None, "modelo": None,
        "cilindros": None, "puertas": None, "pasajeros": None,
        "vin": None, "numero_motor": None, "color": None,
        "transmision": None, "clave_vehicular": None, "nci": None,
    },
    "propietario": {
        "curp": None, "nombre_completo": None, "rfc": None,
        "domicilio": None, "telefono": None, "fecha_nacimiento": None,
    },
    "factura": {
        "numero_factura": None, "fecha": None, "importe": None,
        "uuid": None, "rfc_emisor": None, "rfc_receptor": None,
    },
    "pedimento": {
        "numero_pedimento": None, "tipo_operacion": None,
        "fecha_pago": None, "aduana": None,
    },
}


def _load_prompt(doc_type: str) -> str:
    """Load the type-specific prompt."""
    filename = PROMPT_MAP.get(doc_type, "base.txt")
    prompt_path = PROMPTS_DIR / filename
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()
    # Fallback to base prompt
    base_path = PROMPTS_DIR / "base.txt"
    return base_path.read_text(encoding="utf-8").strip() if base_path.exists() else ""


def _get_client() -> OpenAI:
    """Initialize OpenAI client."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    return OpenAI(api_key=api_key)


def extract_document(image_b64: str, doc_type: str, mime_type: str = "image/jpeg") -> dict:
    """
    Extract structured data from a document image using GPT-4 Vision.

    Args:
        image_b64: Base64-encoded image
        doc_type: Document type (ine, curp, factura, etc.)
        mime_type: MIME type of the image

    Returns:
        Dictionary with extracted fields
    """
    client = _get_client()
    prompt = _load_prompt(doc_type)

    # Determine media type for the API
    if "pdf" in mime_type:
        media_type = "image/png"  # PDFs are converted to PNG
    elif "png" in mime_type:
        media_type = "image/png"
    else:
        media_type = "image/jpeg"

    system_msg = f"""You are a document data extraction specialist for Mexican vehicle documents.
Extract ALL relevant data from this {doc_type.upper()} document.
Return ONLY valid JSON matching this schema — no extra text, no markdown.

{prompt}

Required JSON structure:
{json.dumps(BASE_SCHEMA, indent=2, ensure_ascii=False)}

Rules:
- Extract exactly what you see, don't guess or infer
- Use null for fields not found in the document
- Names in UPPERCASE
- Dates in DD/MM/YYYY format
- VIN/serial numbers exactly as shown (alphanumeric, 17 chars typical)
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Extract all data from this {doc_type} document:"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{image_b64}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
        max_tokens=2000,
        temperature=0.1,
    )

    raw = response.choices[0].message.content.strip()

    # Clean markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        log.warning(f"Failed to parse JSON from GPT response: {raw[:200]}")
        data = {"error": "Failed to parse extraction result", "raw": raw[:500]}

    # Add metadata
    data["tipo_documento"] = doc_type

    log.info(f"Extracted {doc_type}: {len(json.dumps(data))} chars")
    return data
