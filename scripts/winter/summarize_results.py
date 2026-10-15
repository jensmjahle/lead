#!/usr/bin/env python3
"""Summarize and compare closed-loop evaluation runs from run_parallel.py.

For every route the first attempt that was not an infrastructure failure is
used (from results.jsonl), so retries cannot inflate the score; use
--attempt final for the last record instead. Each route is classified as
    ok        status Completed or Perfect
    driving   any other driving failure, including the Bench2Drive game-time
              cap (status "Failed - TickRuntime", 4000 ticks = 200 s)
    infra     simulator crash, agent crash or setup failure, invalid sensors,
              or no result at all
and three driving scores are reported:
    DS_all    mean over all routes, infra routes count as 0. This matches how
              a benchmark score over the full route set is usually reported.
    DS_lead   mean over all routes with a record, using each record's score
              (what LEAD's result_parser averages).
    DS_valid  mean over ok + driving routes, infra routes excluded.

With two or more runs, routes valid in every run are compared pairwise
against the first run: mean paired difference, a bootstrap 95 % CI, and the
share of routes that got worse / better.

Example:
    python scripts/winter/summarize_results.py outputs/eval/b2d_seed0
    python scripts/winter/summarize_results.py outputs/eval/b2d_seed0 outputs/eval/b2d_seed0_snow_medium
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

INFRA_MARKERS = (
    "Simulation crashed",
    "Agent's sensors were invalid",
    "Agent couldn't be set up",
    "Agent crashed",
)


def load_run(run_dir: Path, attempt: str = "first") -> dict[str, dict]:
    """Per-route record. attempt="first": the first attempt that was not an
    infrastructure failure, taken from results.jsonl (retries of driving
    outcomes would otherwise give a best-of-n bias). attempt="final": the record
    left in checkpoint_endpoint.json by the last attempt."""
    results = run_dir / "results.jsonl"
    if attempt == "first" and results.exists():
        routes: dict[str, dict] = {}
        attempts: dict[str, list] = {}
        for line in results.read_text().splitlines():
            if line.strip():
                e = json.loads(line)
                attempts.setdefault(e["route"], []).append(e)
        for route, entries in attempts.items():
            chosen = None
            for e in sorted(entries, key=lambda x: x["attempt"]):
                rec = None if e.get("status") is None else {
                    "status": e["status"], "scores": e["scores"],
                    "meta": {"duration_game": e.get("duration_game")}, "attempt": e["attempt"]}
                if classify(rec) != "infra":
                    chosen = rec
                    break
            routes[route] = chosen
        return routes
    routes = {}
    for ckpt in sorted(run_dir.glob("*/checkpoint_endpoint.json")):
        route = ckpt.parent.name
        try:
            records = json.loads(ckpt.read_text())["_checkpoint"]["records"]
            rec = records[0] if records else None
        except (json.JSONDecodeError, KeyError, IndexError):
            rec = None
        routes[route] = rec
    # Routes that were attempted but never wrote a checkpoint.
    results = run_dir / "results.jsonl"
    if results.exists():
        for line in results.read_text().splitlines():
            if line.strip():
                routes.setdefault(json.loads(line)["route"], None)
    return routes


def classify(rec: dict | None) -> str:
    if rec is None:
        return "infra"
    status = rec.get("status", "")
    if status in ("", "Started") or any(m in status for m in INFRA_MARKERS):
        return "infra"
    if status in ("Completed", "Perfect"):
        return "ok"
    return "driving"


def summarize(name: str, routes: dict[str, dict], expected: int | None) -> dict:
    n = expected or len(routes)
    cls = {r: classify(rec) for r, rec in routes.items()}
    ds = {r: (rec or {}).get("scores", {}).get("score_composed", 0.0) or 0.0 for r, rec in routes.items()}
    rc = {r: (rec or {}).get("scores", {}).get("score_route", 0.0) or 0.0 for r, rec in routes.items()}
    valid = [r for r in routes if cls[r] != "infra"]
    with_record = [r for r in routes if routes[r] is not None]
    statuses: dict[str, int] = {}
    for r, rec in routes.items():
        key = "missing" if rec is None else rec.get("status", "")
        statuses[key] = statuses.get(key, 0) + 1
    summary = {
        "run": name,
        "routes_expected": n,
        "routes_seen": len(routes),
        "ok": sum(c == "ok" for c in cls.values()),
        "driving_failures": sum(c == "driving" for c in cls.values()),
        "infra_failures": sum(c == "infra" for c in cls.values()) + max(0, n - len(routes)),
        "DS_all": sum(ds[r] for r in valid) / n if n else float("nan"),
        "DS_lead": float(np.mean([ds[r] for r in with_record])) if with_record else float("nan"),
        "DS_valid": float(np.mean([ds[r] for r in valid])) if valid else float("nan"),
        "RC_valid": float(np.mean([rc[r] for r in valid])) if valid else float("nan"),
        "statuses": dict(sorted(statuses.items(), key=lambda kv: -kv[1])),
    }
    return summary


def paired(base: dict[str, dict], other: dict[str, dict], n_boot: int = 10000, seed: int = 0) -> dict:
    common = [r for r in base if r in other and classify(base[r]) != "infra" and classify(other[r]) != "infra"]
    if not common:
        return {"n": 0}
    a = np.array([base[r]["scores"]["score_composed"] for r in common])
    b = np.array([other[r]["scores"]["score_composed"] for r in common])
    d = b - a
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "n": len(common),
        "mean_base": float(a.mean()),
        "mean_other": float(b.mean()),
        "mean_diff": float(d.mean()),
        "ci95": [float(lo), float(hi)],
        "worse": int((d < -1e-6).sum()),
        "better": int((d > 1e-6).sum()),
        "equal": int((np.abs(d) <= 1e-6).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--expected", type=int, default=None, help="Total routes in the benchmark (e.g. 220)")
    parser.add_argument("--csv", type=Path, default=None, help="Write per-route table here")
    parser.add_argument("--attempt", choices=("first", "final"), default="first",
                        help="first: first non-infrastructure attempt (default, unbiased); "
                             "final: last record in checkpoint_endpoint.json")
    args = parser.parse_args()

    loaded = [(run.name, load_run(run, args.attempt)) for run in args.runs]
    summaries = [summarize(name, routes, args.expected) for name, routes in loaded]
    for s in summaries:
        print(f"== {s['run']}")
        print(f"   routes {s['routes_seen']}/{s['routes_expected']}  ok {s['ok']}  "
              f"driving failures {s['driving_failures']}  infra failures {s['infra_failures']}")
        print(f"   DS_all {s['DS_all']:.2f}   DS_lead {s['DS_lead']:.2f}   DS_valid {s['DS_valid']:.2f}   "
              f"RC_valid {s['RC_valid']:.2f}")
        print("   statuses: " + ", ".join(f"{k} x{v}" for k, v in s["statuses"].items()))

    base_name, base = loaded[0]
    comparisons = []
    for name, routes in loaded[1:]:
        p = paired(base, routes)
        p["base"], p["other"] = base_name, name
        comparisons.append(p)
        if p["n"]:
            print(f"== paired {name} vs {base_name}: n={p['n']}  dDS={p['mean_diff']:+.2f} "
                  f"[{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}]  worse {p['worse']}  better {p['better']}  "
                  f"equal {p['equal']}")

    out = args.runs[0] / "summary.json" if len(args.runs) == 1 else None
    if out:
        out.write_text(json.dumps({"summaries": summaries, "comparisons": comparisons}, indent=2))
        print(f"wrote {out}")

    if args.csv:
        all_routes = sorted({r for _, routes in loaded for r in routes}, key=lambda r: (len(r), r))
        with open(args.csv, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["route"] + [f"{name}_{col}" for name, _ in loaded for col in ("status", "class", "DS", "RC", "game_s")])
            for r in all_routes:
                row = [r]
                for _, routes in loaded:
                    rec = routes.get(r)
                    if rec is None:
                        row += ["missing" if r in routes else "", "infra" if r in routes else "", "", "", ""]
                    else:
                        row += [rec.get("status"), classify(rec), rec["scores"]["score_composed"],
                                rec["scores"]["score_route"], rec.get("meta", {}).get("duration_game")]
                writer.writerow(row)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
