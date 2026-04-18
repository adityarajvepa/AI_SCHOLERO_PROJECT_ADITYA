"""Identifier helpers for chunks and lectures."""

import re
import unicodedata


def slugify_filename_stem(stem: str) -> str:
    """Convert a filename stem into a stable lecture_id slug."""
    normalized = unicodedata.normalize("NFKD", stem)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    lowered = ascii_only.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "_", lowered)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "lecture"


def make_chunk_id(
    course_id: str,
    lecture_id: str,
    unit_type: str,
    unit_number: int,
    chunk_index: int,
) -> str:
    """Build deterministic chunk_id matching the documented pattern."""
    ut = "page" if unit_type == "page" else "slide"
    return f"{course_id}__{lecture_id}__{ut}_{unit_number}__chunk_{chunk_index}"
