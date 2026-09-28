"""Load facts/policies.yaml, format values by unit, and record which facts a text uses.

Every generated article and reference answer pulls its numbers through `F.v(...)`
(or one of the shortcuts), so the text can't drift from the facts file. While a
`F.recording()` block is open, every key that was read is collected. The corpus
builder stores those keys per article and the question builder uses them to
find gold articles.
"""

from __future__ import annotations

import contextlib
import datetime as dt
from pathlib import Path
from typing import Any, Iterator

import yaml

ROOT = Path(__file__).resolve().parent.parent
FACTS_PATH = ROOT / "facts" / "policies.yaml"

PLANS = ["basic", "plus", "premium"]
PLAN_NAME = {"basic": "Basic", "plus": "Plus", "premium": "Premium"}


def load_facts(path: Path = FACTS_PATH) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def get_path(data: dict, path: str) -> Any:
    cur: Any = data
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            raise KeyError(f"fact path not found: {path}")
    return cur


def money(x: float | int) -> str:
    x = float(x)
    if x >= 100 and x.is_integer():
        return f"${x:,.0f}"
    return f"${x:,.2f}"


def pct(x: float | int, decimals: int | None = None) -> str:
    x = float(x)
    if decimals is not None:
        return f"{x:.{decimals}f}%"
    if x.is_integer():
        return f"{int(x)}%"
    return f"{x:g}%"


def days(n: int, kind: str = "") -> str:
    unit = f"{kind} day" if kind else "day"
    return f"{n} {unit}" + ("" if n == 1 else "s")


def iter_leaves(data: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(data, dict):
        for k, v in data.items():
            yield from iter_leaves(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(data, list):
        for i, v in enumerate(data):
            yield from iter_leaves(v, f"{prefix}.{i}")
    else:
        yield prefix, data


def typed_values(data: dict) -> dict[str, set]:
    """Collect every numeric fact by unit, using the key suffix convention."""
    out: dict[str, set] = {
        "usd": set(),
        "pct": set(),
        "days": set(),
        "business_days": set(),
        "hours": set(),
        "time_et": set(),
    }
    for path, val in iter_leaves(data):
        key = path.split(".")[-1]
        if val is None:
            continue
        if key.endswith("_usd") and isinstance(val, (int, float)):
            out["usd"].add(round(float(val), 2))
        elif key.endswith("_pct") and isinstance(val, (int, float)):
            out["pct"].add(round(float(val), 2))
        elif key.endswith("_business_days") and isinstance(val, int):
            out["business_days"].add(val)
        elif key.endswith("_days") and isinstance(val, int):
            out["days"].add(val)
        elif key.endswith("_hours") and isinstance(val, int):
            out["hours"].add(val)
        elif key.endswith("_time_et") and isinstance(val, str):
            out["time_et"].add(val.strip().upper())
    return out


class Facts:
    def __init__(self, data: dict | None = None):
        self.data = data if data is not None else load_facts()
        self._stack: list[set[str]] = []

    # ------------------------------------------------------------ recording
    @contextlib.contextmanager
    def recording(self) -> Iterator[set[str]]:
        keys: set[str] = set()
        self._stack.append(keys)
        try:
            yield keys
        finally:
            self._stack.pop()

    def mark(self, *keys: str) -> str:
        """Record a key (fact path or 'tag:...') without printing anything."""
        for k in keys:
            if not k.startswith("tag:"):
                get_path(self.data, k)  # fail fast on typos
            for s in self._stack:
                s.add(k)
        return ""

    # ------------------------------------------------------------ raw access
    def raw(self, path: str) -> Any:
        val = get_path(self.data, path)
        self.mark(path)
        return val

    def v(self, path: str, **kw: Any) -> str:
        """Formatted value, unit taken from the key suffix."""
        val = self.raw(path)
        key = path.split(".")[-1]
        if val is None:
            raise ValueError(f"fact {path} is null and can't be printed")
        if key.endswith("_usd"):
            return money(val)
        if key.endswith("_pct"):
            return pct(val, kw.get("decimals"))
        if key.endswith("_business_days"):
            return days(val, "business")
        if key.endswith("_days"):
            return days(val)
        if key.endswith("_hours"):
            return f"{val} hours" if val != 1 else "1 hour"
        if key.endswith("_time_et"):
            return f"{val} ET"
        if isinstance(val, dt.date):
            return val.strftime("%B %-d, %Y")
        return str(val)

    def quiet(self, path: str, **kw: Any) -> str:
        """Formatted value that is NOT recorded. For incidental details in a
        reference answer that shouldn't pull extra articles into its gold set."""
        saved, self._stack = self._stack, []
        try:
            return self.v(path, **kw)
        finally:
            self._stack = saved

    def n(self, path: str) -> Any:
        """Bare number (recorded), for use inside ranges such as '7 to 10 business days'."""
        return self.raw(path)

    # ------------------------------------------------------------ shortcuts
    def fee_path(self, plan: str, name: str, ver: int = 2) -> str:
        return f"fees.versions.v{ver}.{plan}.{name}"

    def fee(self, plan: str, name: str, ver: int = 2) -> str:
        return self.v(self.fee_path(plan, name, ver))

    def fee_raw(self, plan: str, name: str, ver: int = 2) -> Any:
        return self.raw(self.fee_path(plan, name, ver))

    def fee_or_free(self, plan: str, name: str, ver: int = 2) -> str:
        val = self.fee_raw(plan, name, ver)
        return "free" if float(val) == 0 else money(val)

    def lim(self, plan: str, name: str) -> str:
        return self.v(f"limits.{plan}.{name}")

    def date(self, path: str) -> str:
        val = self.raw(path)
        if isinstance(val, str):
            val = dt.date.fromisoformat(val)
        return val.strftime("%B %-d, %Y")

    def iso(self, path: str) -> str:
        val = self.raw(path)
        return val.isoformat() if isinstance(val, dt.date) else str(val)

    def range_bd(self, lo: str, hi: str) -> str:
        return f"{self.n(lo)} to {self.n(hi)} business days"


def fact_keys_for_value_check() -> dict[str, set]:
    return typed_values(load_facts())
