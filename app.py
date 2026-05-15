"""
AI Document Extractor — Flask API
Extracts structured data from Mexican vehicle documents using GPT-4 Vision.
"""
import os
import json
import base64
import logging
from flask import Flask, request, jsonify
from functools import wraps

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024  # 25MB

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("extractor")

API_KEY = os.environ.get("API_KEY", "demo-key")


# ── Auth ──
def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.headers.get("Authorization", "")
        token = auth.replace("Bearer ", "").strip()
        if token != API_KEY:
            return jsonify({"ok": False, "error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated


# ── Extraction Engine ──
from extractor.extractor import extract_document
from extractor.pdf_processor import pdf_to_images
from extractor.classifier import classify_document
from extractor.validator import cross_validate


@app.route("/api/extract", methods=["POST"])
@require_auth
def api_extract():
    """Extract data from a single document."""
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "No file provided"}), 400

    file = request.files["file"]
    filename = file.filename or "unknown"
    content = file.read()
    mime = file.content_type or "application/octet-stream"

    log.info(f"Processing: {filename} ({len(content)} bytes)")

    try:
        # Convert PDF to images if needed
        if mime == "application/pdf" or filename.lower().endswith(".pdf"):
            images = pdf_to_images(content)
            if not images:
                return jsonify({"ok": False, "error": "Could not process PDF"}), 400
            image_data = images[0]  # Use first page
        else:
            image_data = base64.b64encode(content).decode("utf-8")

        # Classify document type
        doc_type = classify_document(image_data, mime)
        log.info(f"Classified as: {doc_type}")

        # Extract structured data
        data = extract_document(image_data, doc_type, mime)
        log.info(f"Extraction complete: {doc_type}")

        return jsonify({
            "ok": True,
            "document_type": doc_type,
            "data": data,
        })

    except Exception as e:
        log.exception(f"Extraction error: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/extract/batch", methods=["POST"])
@require_auth
def api_extract_batch():
    """Extract data from multiple documents and cross-validate."""
    files = request.files.getlist("files[]")
    if not files:
        return jsonify({"ok": False, "error": "No files provided"}), 400

    results = []
    for file in files:
        filename = file.filename or "unknown"
        content = file.read()
        mime = file.content_type or "application/octet-stream"

        try:
            if mime == "application/pdf" or filename.lower().endswith(".pdf"):
                images = pdf_to_images(content)
                image_data = images[0] if images else None
            else:
                image_data = base64.b64encode(content).decode("utf-8")

            if not image_data:
                results.append({"filename": filename, "ok": False, "error": "Could not process"})
                continue

            doc_type = classify_document(image_data, mime)
            data = extract_document(image_data, doc_type, mime)

            results.append({
                "filename": filename,
                "ok": True,
                "document_type": doc_type,
                "data": data,
            })
        except Exception as e:
            results.append({"filename": filename, "ok": False, "error": str(e)})

    # Cross-validate extracted data
    validation = cross_validate(results)

    return jsonify({
        "ok": True,
        "results": results,
        "validation": validation,
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "service": "ai-document-extractor"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
