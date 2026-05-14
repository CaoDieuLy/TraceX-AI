"""Helpers for Vietnamese UI display text.

The database keeps canonical English labels/summaries for scoring. Metadata
service translates only response payloads so persisted search data stays stable.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_QUERY_SERVICE_URL: str | None = None
_EMPTY_MARKERS = {"", "none", "null", "n/a"}
_TRACKLET_TEXT_FIELDS = [
    "gender",
    "age_range",
    "upper_color",
    "upper_type",
    "upper_desc",
    "lower_color",
    "lower_type",
    "lower_desc",
    "shoes_color",
    "shoes_type",
    "shoes_desc",
    "bag_presence",
    "bag_type",
    "bag_desc",
    "hat_presence",
    "hat_color",
    "hat_type",
    "hat_desc",
    "mask_presence",
    "hair_style",
    "hair_color",
]


def _get_query_service_url() -> str:
    global _QUERY_SERVICE_URL
    if _QUERY_SERVICE_URL is None:
        _QUERY_SERVICE_URL = os.getenv("QUERY_SERVICE_URL", "http://query-service:8003")
    return _QUERY_SERVICE_URL.rstrip("/")


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in _EMPTY_MARKERS:
        return None
    return text or None


def _join_parts(values: list[Any]) -> str | None:
    parts: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_text(value)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        parts.append(text)
    if not parts:
        return None
    if all(part.lower() == "unknown" for part in parts):
        return "unknown"
    return " ".join(parts)


def translate_texts_to_vietnamese(texts: list[str | None]) -> dict[str, str]:
    """Return a raw-text → Vietnamese-text map via query-service translation."""
    unique: list[str] = []
    seen: set[str] = set()
    for value in texts:
        text = _clean_text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)

    if not unique:
        return {}

    url = f"{_get_query_service_url()}/api/v1/translate/to-vietnamese"
    try:
        with httpx.Client(timeout=float(os.getenv("TRANSLATION_DISPLAY_TIMEOUT_SECONDS", "90"))) as client:
            response = client.post(url, json={"texts": unique})
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPError as exc:
        logger.warning("Vietnamese display translation failed: %s", exc)
        return {text: text for text in unique}

    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return {text: text for text in unique}

    translated: dict[str, str] = {}
    for raw, item in zip(unique, items):
        translated[raw] = str(item or raw)
    for raw in unique[len(items):]:
        translated[raw] = raw
    return translated


def _tracklet_display_raw(tracklet: dict[str, Any]) -> dict[str, str | None]:
    bag_presence = _clean_text(tracklet.get("bag_presence"))
    hat_presence = _clean_text(tracklet.get("hat_presence"))

    if bag_presence == "yes":
        bag = _join_parts([tracklet.get("bag_type"), tracklet.get("bag_desc")]) or "yes"
    else:
        bag = bag_presence

    if hat_presence == "yes":
        hat = _join_parts([tracklet.get("hat_color"), tracklet.get("hat_type")]) or "yes"
    else:
        hat = hat_presence

    return {
        "gender": _clean_text(tracklet.get("gender")),
        "age_range": _clean_text(tracklet.get("age_range")),
        "hair": _join_parts([tracklet.get("hair_style"), tracklet.get("hair_color")]),
        "upper": _join_parts([tracklet.get("upper_color"), tracklet.get("upper_type")]),
        "lower": _join_parts([tracklet.get("lower_color"), tracklet.get("lower_type")]),
        "shoes": _join_parts([tracklet.get("shoes_color"), tracklet.get("shoes_type")]),
        "bag": bag,
        "hat": hat,
        "mask": _clean_text(tracklet.get("mask_presence")),
    }


def localize_candidate_detail_payload(payload: Any) -> Any:
    """Inject Vietnamese display text into a candidate-detail response payload."""
    if not isinstance(payload, dict):
        return payload

    tracklets = payload.get("tracklets")
    if not isinstance(tracklets, list):
        return payload

    raw_texts: list[str | None] = [_clean_text(payload.get("appearance_summary"))]
    display_by_tracklet: list[dict[str, str | None]] = []

    for tracklet in tracklets:
        if not isinstance(tracklet, dict):
            display_by_tracklet.append({})
            continue
        raw_texts.append(_clean_text(tracklet.get("appearance_summary")))
        for field_name in _TRACKLET_TEXT_FIELDS:
            raw_texts.append(_clean_text(tracklet.get(field_name)))
        display = _tracklet_display_raw(tracklet)
        display_by_tracklet.append(display)
        raw_texts.extend(display.values())

        actions = tracklet.get("actions")
        if isinstance(actions, list):
            for action in actions:
                if isinstance(action, dict):
                    raw_texts.append(_clean_text(action.get("action_label")))
                    raw_texts.append(_clean_text(action.get("kinetics_label")))

    translated = translate_texts_to_vietnamese(raw_texts)

    summary = _clean_text(payload.get("appearance_summary"))
    if summary:
        payload["appearance_summary"] = translated.get(summary, summary)

    for tracklet, display_raw in zip(tracklets, display_by_tracklet):
        if not isinstance(tracklet, dict):
            continue
        summary = _clean_text(tracklet.get("appearance_summary"))
        if summary:
            tracklet["appearance_summary"] = translated.get(summary, summary)
        for field_name in _TRACKLET_TEXT_FIELDS:
            raw_value = _clean_text(tracklet.get(field_name))
            if raw_value:
                tracklet[field_name] = translated.get(raw_value, raw_value)
        tracklet["display"] = {
            key: translated.get(value, value) if value else None
            for key, value in display_raw.items()
        }

        actions = tracklet.get("actions")
        if isinstance(actions, list):
            for action in actions:
                if not isinstance(action, dict):
                    continue
                label = _clean_text(action.get("action_label"))
                kinetics = _clean_text(action.get("kinetics_label"))
                if label:
                    action["action_label_vi"] = translated.get(label, label)
                    action["action_label"] = translated.get(label, label)
                if kinetics:
                    action["kinetics_label_vi"] = translated.get(kinetics, kinetics)
                    action["kinetics_label"] = translated.get(kinetics, kinetics)

    return payload
