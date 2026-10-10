"""Final test evaluation, run once after an evolution run (2026-10-10).

    python final_eval.py <circuit.py> [<circuit.py> ...] [--out final_eval.json]

During a run every circuit is scored on validation only and never sees the 30 test
flowers. This is the one place they are used: each circuit passed here (the final
front, picked by validation, plus the seed) is retrained from the same 10 starts on
the same 90 training flowers through the seed's own train_starts(), then scored once
on test. Report the averages; never use them to choose between circuits.

Every circuit's evolved blocks are put under this directory's seed_vqc.py header, so
circuits from older runs are trained and scored with the current protected code.
"""
import argparse, io, contextlib, json, os, types
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MARK = "# --OPTION--"
HEADER = open(os.path.join(HERE, "seed_vqc.py")).read().split(MARK)[0]


def load(path):
    blocks = open(path).read().replace("\r", "").split(MARK)[1:]
    src = MARK.join([HEADER] + blocks)
    mod = types.ModuleType(os.path.basename(path)[:-3])    # not __main__, so main() doesn't run
    with contextlib.redirect_stdout(io.StringIO()):         # some evolved blocks print at import
        exec(compile(src, path, "exec"), mod.__dict__)
    return mod


def evaluate(path):
    m = load(path)
    X_tr, X_va, X_te, y_tr, y_va, y_te = m.load_data()
    qc, xp, wp = m.build_circuit()
    rows = []
    for seed, passes, w in m.train_starts(qc, xp, wp, X_tr, y_tr):
        rows.append({"seed": seed, "train_passes": passes,
                     "test_accuracy": m.accuracy(qc, xp, wp, w, X_te, y_te),
                     "test_soft_accuracy": m.soft_accuracy(qc, xp, wp, w, X_te, y_te),
                     "test_cross_entropy": m.cross_entropy(qc, xp, wp, w, X_te, y_te),
                     "val_accuracy": m.accuracy(qc, xp, wp, w, X_va, y_va),
                     "weights": [float(v) for v in w]})
    acc = np.array([r["test_accuracy"] for r in rows])
    return {"circuit": os.path.basename(path), "gates": m.gate_count(qc), "n_starts": len(rows),
            "test_accuracy": float(acc.mean()), "test_accuracy_sd": float(acc.std()),
            "test_soft_accuracy": float(np.mean([r["test_soft_accuracy"] for r in rows])),
            "val_accuracy": float(np.mean([r["val_accuracy"] for r in rows])),
            # every start ending on identical weights means the training block ignores SEED
            "fixed_start": all(np.allclose(r["weights"], rows[0]["weights"]) for r in rows),
            "starts": rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("circuits", nargs="+")
    ap.add_argument("--out", default="final_eval.json")
    args = ap.parse_args()
    out = []
    print(f"{'circuit':40} {'gates':>5} {'test top-1':>11} {'sd':>6} {'test soft':>10} {'val top-1':>10}")
    for p in args.circuits:
        r = evaluate(p); out.append(r)
        print(f"{r['circuit'][:40]:40} {r['gates']:5.0f} {r['test_accuracy']:11.1%} {r['test_accuracy_sd']:6.1%} "
              f"{r['test_soft_accuracy']:10.1%} {r['val_accuracy']:10.1%}"
              + ("   FIXED START: one training repeated" if r["fixed_start"] else ""))
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"-> {args.out}")
