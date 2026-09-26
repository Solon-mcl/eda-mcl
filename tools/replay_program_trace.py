#!/usr/bin/env python3
"""Replay compressed macro traces and run cycle-preserving deletion tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run_experiments import ROOT, load_harness


def _play(dut, backend, records, idle_action, stop_index=None,
          delete_record=None, delete_chunk=None):
    _, harness_cls, _ = load_harness(dut, backend)
    harness = harness_cls(backend=backend)
    state = np.asarray(harness.reset(), dtype=np.float32).reshape(-1)
    limit = len(records) - 1 if stop_index is None else int(stop_index)
    try:
        for record_index, record in enumerate(records[:limit + 1]):
            sequence = list(record.get("action_sequence", ()))
            if record_index == delete_record and delete_chunk is None:
                sequence = [{"action": idle_action,
                             "cycles": sum(int(item.get("cycles", 1))
                                           for item in sequence)}]
            for chunk_index, item in enumerate(sequence):
                action = item.get("action", idle_action)
                if (record_index == delete_record and
                        delete_chunk == chunk_index):
                    action = idle_action
                action = np.asarray(action, dtype=np.float32).reshape(-1)
                for _ in range(max(1, int(item.get("cycles", 1)))):
                    state, _, _, _ = harness.step(action)
                    state = np.asarray(state, dtype=np.float32).reshape(-1)
        return state
    finally:
        if hasattr(harness._dut, "close"):
            harness._dut.close()


def _indices(state):
    return set(np.flatnonzero(np.asarray(state) > 0.5).astype(int).tolist())


def _coverage_bin_names(base):
    path = Path(base) / "dut" / "coverage_meta.json"
    if not path.exists():
        return []
    meta = json.loads(path.read_text(encoding="utf-8"))
    return [f"{point.get('name', '')}.{item.get('name', '')}"
            for point in meta.get("coverpoints", ())
            for item in point.get("bins", ())]


def _named(indices, names):
    return [names[index] if 0 <= index < len(names) else str(index)
            for index in indices]


def analyze(record, max_records):
    traces = list(record.get("replay_trace", ()))
    if not traces:
        raise ValueError("input has no replay_trace; rerun with --trace-actions")
    idle = list(record.get("replay_idle_action", ()))
    dut, backend = record["dut"], record["backend"]
    _, _, base = load_harness(dut, backend)
    bin_names = _coverage_bin_names(base)
    full_state = _play(dut, backend, traces, idle)
    productive = [
        (index, item) for index, item in enumerate(traces)
        if int(item.get("new_bins", 0)) > 0 and item.get("action_sequence")
    ]
    productive.sort(key=lambda pair: (-int(pair[1].get("new_bins", 0)),
                                      pair[0]))
    selected = productive[:max(0, int(max_records))]
    analyses = []
    for index, item in selected:
        baseline = _indices(_play(dut, backend, traces, idle,
                                  stop_index=index))
        without = _indices(_play(dut, backend, traces, idle,
                                 stop_index=index, delete_record=index))
        chunk_results = []
        for chunk_index, chunk in enumerate(item["action_sequence"]):
            deleted = _indices(_play(
                dut, backend, traces, idle, stop_index=index,
                delete_record=index, delete_chunk=chunk_index))
            essential = sorted(baseline - deleted)
            if essential:
                chunk_results.append({
                    "chunk_index": chunk_index,
                    "cycles": int(chunk.get("cycles", 1)),
                    "essential_bin_indices": essential,
                    "essential_bins": _named(essential, bin_names),
                })
        claimed = list(item.get("new_bin_indices", ()))
        causal = sorted(baseline - without)
        analyses.append({
            "record_index": index,
            "macro": item.get("macro"),
            "cycles": int(item.get("cycles", 0)),
            "claimed_new_bin_indices": claimed,
            "claimed_new_bins": _named(claimed, bin_names),
            "causal_bin_indices": causal,
            "causal_bins": _named(causal, bin_names),
            "essential_chunks": chunk_results,
            "context": dict(item.get("context", {})),
        })
    return {
        "dut": dut,
        "backend": backend,
        "source_seed": int(record.get("seed", 0)),
        "source_steps": int(record.get("steps", 0)),
        "source_covered_bins": int(record.get("covered_bins", 0)),
        "replayed_trace_records": len(traces),
        "full_trace_replay_covered_bins": int(np.sum(full_state > 0.5)),
        "productive_records": len(productive),
        "analyzed_records": len(analyses),
        "records": analyses,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--record", type=int, default=0)
    parser.add_argument("--max-records", type=int, default=12)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = json.loads(Path(args.input).read_text(encoding="utf-8"))
    records = source if isinstance(source, list) else [source]
    report = analyze(records[args.record], args.max_records)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
