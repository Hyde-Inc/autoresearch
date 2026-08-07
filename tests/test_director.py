import pytest

from autoresearch.director import parse_json_object


def test_parses_plain_json() -> None:
    assert parse_json_object('{"ideas": [{"title": "ARIMA"}]}')["ideas"][0]["title"] == "ARIMA"


def test_parses_fenced_and_prefixed_json() -> None:
    fenced = 'Here are the ideas:\n```json\n{"ideas": [{"title": "XGBoost"}]}\n```'
    assert parse_json_object(fenced)["ideas"][0]["title"] == "XGBoost"


def test_rejects_response_without_json() -> None:
    with pytest.raises(ValueError, match="did not return a JSON object"):
        parse_json_object("I would try a transformer.")
