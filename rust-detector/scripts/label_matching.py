"""Country and date matching for anomaly labels."""

from datetime import date
from typing import Any

def label_country_code(label: dict[str, Any]) -> str | None:
    code = label.get("country_code")
    legacy = label.get("metadata", {}).get("country_code")
    for value in (code, legacy):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError("label country code must be a nonempty string")
    if code is not None and legacy is not None and code.strip().upper() != legacy.strip().upper():
        raise ValueError("label country codes disagree between top level and metadata")
    value = code if code is not None else legacy
    return value.strip().upper() if value is not None else None


def label_applies(label: dict[str, Any], country: str) -> bool:
    if label["scope"] == "country":
        code = label_country_code(label)
        additional = [
            code.upper() for code in label.get("additional_matching_countries", [])
        ]
        return code is None or code.upper() == country or country in additional
    affected = [code.upper() for code in label.get("affected_countries", [])]
    return not affected or country in affected


def label_matches(label: dict[str, Any], anomaly_date: date, proximity_days: int) -> bool:
    distance = (anomaly_date - date.fromisoformat(label["date"])).days
    if label.get("match_mode", "proximity") == "exact":
        return distance == 0
    window = min(proximity_days, 1) if label["scope"] == "global" else proximity_days
    return abs(distance) <= window


def matched_label(
    anomaly: dict[str, Any],
    labels: list[dict[str, Any]],
    country: str,
    proximity_days: int,
) -> dict[str, Any] | None:
    anomaly_date = date.fromisoformat(anomaly["start"])
    candidates = [
        (label, (anomaly_date - date.fromisoformat(label["date"])).days, index)
        for index, label in enumerate(labels)
        if label_applies(label, country)
        and label_matches(label, anomaly_date, proximity_days)
    ]
    if not candidates:
        return None
    label, distance, _ = min(
        candidates,
        key=lambda candidate: (
            abs(candidate[1]),
            candidate[0]["scope"] == "global",
            candidate[0]["date"],
            candidate[2],
        ),
    )
    return {"label": label, "distance_days": distance}
