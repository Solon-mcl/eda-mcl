#!/usr/bin/env python3
"""Fit a conservative macro Q head from measured RTL trajectories.

The permutation-invariant Deep-Sets encoder is retained from the synthetic
pretraining stage.  Its final four-action head is refit with semi-Markov
Bellman targets derived from real macro durations and coverage gains.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from inference.coverage_controller import element_features, load_bin_descriptors  # noqa: E402

META = {
    "spi_master_public": ROOT / "public_duts/spi_master_public/spi_master_public/dut/covergroup.svh",
    "spi_xfer_public": ROOT / "public_duts/spi_xfer_public/spi_xfer_public/dut/covergroup.svh",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/macro_trajectories_verilator.json")
    parser.add_argument("--base", default="app/inference/model/coverage_controller.npz")
    parser.add_argument("--output", default="app/inference/model/coverage_q_controller.npz")
    parser.add_argument("--metrics", default="results/coverage_q_training.json")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--gamma-per-1k", type=float, default=0.97)
    parser.add_argument("--cql-alpha", type=float, default=0.04)
    args = parser.parse_args()

    payload = json.loads((ROOT / args.input).read_text(encoding="utf-8"))
    base = np.load(ROOT / args.base, allow_pickle=False)
    phi_w, phi_b = base["phi_w"], base["phi_b"]
    rho_w1, rho_b1 = base["rho_w1"], base["rho_b1"]
    initial_w2, initial_b2 = base["rho_w2"], base["rho_b2"]
    descriptors = {name: load_bin_descriptors(str(path))
                   for name, path in META.items()}

    def hidden(dut, state, step, steps):
        state = np.asarray(state, dtype=np.float32)
        progress = float(step) / max(1, int(steps) - 1)
        elements = element_features(state, descriptors[dut], progress)
        emb = np.maximum(elements @ phi_w + phi_b, 0.0)
        all_pool = emb.mean(axis=0)
        missing = state < 0.5
        miss_pool = (emb[missing].mean(axis=0) if np.any(missing)
                     else np.zeros(emb.shape[1], dtype=np.float32))
        glob = np.asarray([float(np.mean(state)), progress, 0.0], dtype=np.float32)
        pooled = np.concatenate([all_pool, miss_pool, glob])
        return np.maximum(pooled @ rho_w1 + rho_b1, 0.0)

    samples = []
    initial_states = {}
    for record in payload["records"]:
        order = record["order"]
        steps = record["steps"]
        total = record["total_bins"]
        for position, transition in enumerate(record["transitions"]):
            dut = transition["dut"]
            h = hidden(dut, transition["state"], transition["step"], steps)
            next_step = transition["step"] + transition["duration"]
            hn = hidden(dut, transition["next_state"], next_step, steps)
            duration_k = transition["duration"] / 1000.0
            discount = args.gamma_per_1k ** duration_k
            # Approximate the gain as arriving halfway through the macro.
            reward = (transition["reward_bins"] / total) * np.sqrt(discount)
            samples.append({
                "h": h, "hn": hn, "action": int(transition["action"]),
                "reward": float(reward), "discount": float(discount),
                "remaining": [int(a) for a in order[position + 1:]],
            })
            if transition["step"] == 0:
                initial_states[dut] = h

    width = initial_w2.shape[0]
    w2 = initial_w2.astype(np.float64).copy()
    b2 = initial_b2.astype(np.float64).copy()
    ridge = 2e-3
    for _ in range(max(1, args.iterations)):
        old_w, old_b = w2.copy(), b2.copy()
        targets = []
        for sample in samples:
            future = 0.0
            if sample["remaining"]:
                next_q = sample["hn"] @ old_w + old_b
                future = max(float(next_q[a]) for a in sample["remaining"])
            targets.append(sample["reward"] + sample["discount"] * future)
        for action in range(4):
            rows, values, weights = [], [], []
            for sample, target in zip(samples, targets):
                row = np.append(sample["h"], 1.0)
                if sample["action"] == action:
                    rows.append(row); values.append(target); weights.append(1.0)
                else:
                    # CQL-style conservative pseudo-observation: unsupported
                    # actions at this exact state are softly pulled toward 0.
                    rows.append(row); values.append(0.0); weights.append(args.cql_alpha)
            x = np.asarray(rows, dtype=np.float64)
            y = np.asarray(values, dtype=np.float64)
            sw = np.sqrt(np.asarray(weights, dtype=np.float64))
            xw, yw = x * sw[:, None], y * sw
            reg = np.eye(width + 1) * ridge
            reg[-1, -1] = ridge * 0.1
            coef = np.linalg.solve(xw.T @ xw + reg, xw.T @ yw)
            w2[:, action], b2[action] = coef[:-1], coef[-1]

    errors = []
    for sample in samples:
        q = sample["h"] @ w2 + b2
        future = 0.0
        if sample["remaining"]:
            nq = sample["hn"] @ w2 + b2
            future = max(float(nq[a]) for a in sample["remaining"])
        target = sample["reward"] + sample["discount"] * future
        errors.append(float(q[sample["action"]] - target))

    initial_q = {dut: (h @ w2 + b2).tolist()
                 for dut, h in initial_states.items()}
    metrics = {
        "method": "fitted Q iteration with semi-Markov discount and conservative pseudo-observations",
        "source_backend": payload["backend"],
        "transitions": len(samples),
        "iterations": args.iterations,
        "gamma_per_1000_cycles": args.gamma_per_1k,
        "cql_alpha": args.cql_alpha,
        "bellman_rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "initial_q": initial_q,
        "initial_greedy_macro": {dut: int(np.argmax(q)) for dut, q in initial_q.items()},
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, phi_w=phi_w, phi_b=phi_b,
                        rho_w1=rho_w1, rho_b1=rho_b1,
                        rho_w2=w2.astype(np.float32), rho_b2=b2.astype(np.float32))
    metrics_path = ROOT / args.metrics
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
