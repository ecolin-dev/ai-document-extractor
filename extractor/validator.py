"""
Cross-document validation.
Compares extracted data across multiple documents to catch inconsistencies.
"""
import logging
from typing import List, Dict

log = logging.getLogger("extractor.validator")


def cross_validate(results: List[Dict]) -> Dict:
    """
    Cross-validate extracted data from multiple documents.

    Checks:
    - VIN consistency across invoice and REPUVE
    - Name consistency across INE and CURP
    - RFC/CURP format validation

    Args:
        results: List of extraction results from batch processing

    Returns:
        Validation report with matches and mismatches
    """
    successful = [r for r in results if r.get("ok") and r.get("data")]
    if len(successful) < 2:
        return {"status": "skipped", "reason": "Need at least 2 documents to validate"}

    # Collect all extracted values by field
    vins = []
    names = []
    curps = []
    rfcs = []

    for r in successful:
        data = r["data"]
        veh = data.get("vehiculo", {}) or {}
        prop = data.get("propietario", {}) or {}

        if veh.get("vin"):
            vins.append({"source": r["document_type"], "value": veh["vin"].strip().upper()})
        if prop.get("nombre_completo"):
            names.append({"source": r["document_type"], "value": prop["nombre_completo"].strip().upper()})
        if prop.get("curp"):
            curps.append({"source": r["document_type"], "value": prop["curp"].strip().upper()})
        if prop.get("rfc"):
            rfcs.append({"source": r["document_type"], "value": prop["rfc"].strip().upper()})

    report = {"status": "ok", "checks": []}

    # VIN check
    if len(vins) >= 2:
        unique_vins = set(v["value"] for v in vins)
        report["checks"].append({
            "field": "VIN",
            "match": len(unique_vins) == 1,
            "values": vins,
            "note": "All documents show same VIN" if len(unique_vins) == 1 else "VIN mismatch detected",
        })

    # Name check (fuzzy — names may have slight differences)
    if len(names) >= 2:
        match = _fuzzy_name_match(names)
        report["checks"].append({
            "field": "nombre",
            "match": match,
            "values": names,
            "note": "Names are consistent" if match else "Name discrepancy — may be due to owner change",
        })

    # CURP format validation
    for c in curps:
        valid = _validate_curp(c["value"])
        report["checks"].append({
            "field": "CURP",
            "match": valid,
            "values": [c],
            "note": f"CURP format {'valid' if valid else 'invalid'}: {c['value']}",
        })

    # RFC format validation
    for r_val in rfcs:
        valid = _validate_rfc(r_val["value"])
        report["checks"].append({
            "field": "RFC",
            "match": valid,
            "values": [r_val],
            "note": f"RFC format {'valid' if valid else 'invalid'}: {r_val['value']}",
        })

    # Overall status
    mismatches = [c for c in report["checks"] if not c["match"]]
    if mismatches:
        report["status"] = "warnings"
        report["warning_count"] = len(mismatches)

    return report


def _fuzzy_name_match(names: List[Dict]) -> bool:
    """Check if names are similar enough (handles word order differences)."""
    if len(names) < 2:
        return True

    def normalize(name: str) -> set:
        return set(name.upper().replace(",", "").split())

    base = normalize(names[0]["value"])
    for n in names[1:]:
        other = normalize(n["value"])
        # At least 2 words must match
        common = base & other
        if len(common) < 2:
            return False
    return True


def _validate_curp(curp: str) -> bool:
    """Basic CURP format validation (18 alphanumeric characters)."""
    import re
    return bool(re.match(r"^[A-Z]{4}\d{6}[HM][A-Z]{5}[A-Z0-9]\d$", curp))


def _validate_rfc(rfc: str) -> bool:
    """Basic RFC format validation (12-13 alphanumeric characters)."""
    import re
    return bool(re.match(r"^[A-ZÑ&]{3,4}\d{6}[A-Z0-9]{3}$", rfc))
