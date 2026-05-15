"""
Tests for the document extractor.
Run: pytest tests/ -v
"""
import json
import pytest
from extractor.validator import _validate_curp, _validate_rfc, _fuzzy_name_match, cross_validate


class TestCURPValidation:
    def test_valid_curp(self):
        assert _validate_curp("DIHA000314HPLZRNA4") == True

    def test_invalid_curp_too_short(self):
        assert _validate_curp("DIHA000314") == False

    def test_invalid_curp_bad_format(self):
        assert _validate_curp("1234567890ABCDEFGH") == False


class TestRFCValidation:
    def test_valid_rfc_persona_fisica(self):
        assert _validate_rfc("DIHA000314A5B") == True

    def test_valid_rfc_persona_moral(self):
        assert _validate_rfc("DAM211215CA7") == True

    def test_generic_rfc(self):
        assert _validate_rfc("XAXX010101000") == True

    def test_invalid_rfc(self):
        assert _validate_rfc("123") == False


class TestFuzzyNameMatch:
    def test_same_name(self):
        names = [
            {"source": "ine", "value": "ANDRES DIAZ HERNANDEZ"},
            {"source": "curp", "value": "ANDRES DIAZ HERNANDEZ"},
        ]
        assert _fuzzy_name_match(names) == True

    def test_reordered_name(self):
        names = [
            {"source": "ine", "value": "DIAZ HERNANDEZ ANDRES"},
            {"source": "factura", "value": "ANDRES DIAZ HERNANDEZ"},
        ]
        assert _fuzzy_name_match(names) == True

    def test_different_names(self):
        names = [
            {"source": "ine", "value": "ANDRES DIAZ HERNANDEZ"},
            {"source": "factura", "value": "MARIA LOPEZ GARCIA"},
        ]
        assert _fuzzy_name_match(names) == False


class TestCrossValidation:
    def test_matching_vins(self):
        results = [
            {
                "ok": True,
                "document_type": "factura",
                "data": {"vehiculo": {"vin": "3N1EB31S11K344337"}, "propietario": {}},
            },
            {
                "ok": True,
                "document_type": "repuve",
                "data": {"vehiculo": {"vin": "3N1EB31S11K344337"}, "propietario": {}},
            },
        ]
        report = cross_validate(results)
        vin_check = next(c for c in report["checks"] if c["field"] == "VIN")
        assert vin_check["match"] == True

    def test_mismatched_vins(self):
        results = [
            {
                "ok": True,
                "document_type": "factura",
                "data": {"vehiculo": {"vin": "3N1EB31S11K344337"}, "propietario": {}},
            },
            {
                "ok": True,
                "document_type": "repuve",
                "data": {"vehiculo": {"vin": "1FTCR11XXRUA30343"}, "propietario": {}},
            },
        ]
        report = cross_validate(results)
        vin_check = next(c for c in report["checks"] if c["field"] == "VIN")
        assert vin_check["match"] == False

    def test_insufficient_documents(self):
        results = [{"ok": True, "document_type": "ine", "data": {"vehiculo": {}, "propietario": {}}}]
        report = cross_validate(results)
        assert report["status"] == "skipped"
