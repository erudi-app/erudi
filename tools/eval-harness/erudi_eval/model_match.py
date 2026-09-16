"""Find the installed row of a catalog model.

After a download Erudi rewrites the row's `link` to the local model directory, so matching on the
catalog link alone never finds an installed model and the harness downloads a second copy (observed
on 2026-09-15: catalog row 20 `lmstudio-community/Qwen3-4B-MLX-4bit`, installed rows 358 and 360 with
links `.../data/models/358`). The catalog row's `name` is the bridge between the two shapes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ModelMatchError(RuntimeError):
    """The catalog model is present in a state the harness must not measure or download over."""


@dataclass
class Match:
    model: dict[str, Any]
    rule: str  # "link" or "name"
    candidates: int
    catalog_name: str | None
    note: str

    @property
    def by_name(self) -> bool:
        return self.rule == "name"


def _is_base_model(row: dict[str, Any]) -> bool:
    return not row.get("is_attached_to_kb")  # KB assistants copy their base model's link and name


def pick_installed(local_rows: list[dict[str, Any]], catalog_rows: list[dict[str, Any]], link: str, data_root: str) -> Match | None:
    """The installed base model for `link`, or None when it is not installed.

    Raises ModelMatchError when a matching row is mid-download or has lost its weights.
    """
    catalog = next((c for c in catalog_rows if c.get("link") == link), None)
    name = (catalog or {}).get("name")
    base = [r for r in local_rows if _is_base_model(r)]
    matched = [r for r in base if r.get("link") == link] or []
    rule = "link"
    if not matched and name:
        matched = [r for r in base if r.get("name") == name]
        rule = "name"
    if not matched:
        return None
    if any(r.get("local") == 2 for r in matched):
        raise ModelMatchError(f"model {link} is being downloaded outside the harness")
    installed = [r for r in matched if r.get("local") == 1]
    if not installed:
        return None
    usable = [r for r in installed if r.get("weights_available") is not False]
    if not usable:
        raise ModelMatchError(f"model {link} is installed but its weights are missing")
    inside = [r for r in usable if str(r.get("link", "")).startswith(str(data_root))]
    chosen = sorted(inside or usable, key=lambda r: r["id"])[0]
    note = f"matched by {rule}"
    if rule == "name":
        note += f" ({name!r}): the installed row's link is the local path, not the catalog link"
    if len(usable) > 1:
        note += f"; {len(usable)} installed rows matched, kept id {chosen['id']}"
    return Match(model=chosen, rule=rule, candidates=len(usable), catalog_name=name, note=note)
