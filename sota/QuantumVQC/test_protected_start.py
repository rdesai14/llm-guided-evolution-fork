"""Checks the protected training start (2026-11-16). Run on PACE, not the Mac (scores differ there):

    srun -A coc -q coe-ice -p pace-cpu -N1 -n1 -c4 --mem=8G -t 20 uv run python sota/QuantumVQC/test_protected_start.py

main() draws each training's starting angles and hands them to train_angles as w0. A block
that pins its own start must be rejected, and a block that sets SEED must change nothing.
"""
import pathlib, re, subprocess, sys, tempfile

SRC = (pathlib.Path(__file__).parent / "network.py").read_text()
EXPECT = "obj1 val error 0.1100 (89.0% acc)   obj2 gates 28"   # the seed on PACE
OWN_START = "    def objective(w):"
VARIANTS = {
    "seed": SRC,
    "pinned": SRC.replace(OWN_START, "    w0 = np.random.default_rng(42).uniform(0, 2 * np.pi, len(w_params))\n" + OWN_START),
    "old_style": SRC.replace(OWN_START, "    w0 = np.random.default_rng(SEED).uniform(0, 2 * np.pi, len(w_params))\n" + OWN_START),
    "seed_rebound": SRC.replace("MAX_ITER = 178", "SEED = 7\nMAX_ITER = 178"),
}
assert all(v != SRC for k, v in VARIANTS.items() if k != "seed"), "a variant did not apply"

with tempfile.TemporaryDirectory() as d:
    procs = {}
    for name, src in VARIANTS.items():
        path = pathlib.Path(d) / f"network_{name}.py"
        path.write_text(src)
        procs[name] = subprocess.Popen([sys.executable, str(path), "--gene-id", name, "--out-dir", d],
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = {name: p.communicate()[0] for name, p in procs.items()}

score = {name: (re.search(r"obj1 val error .*?obj2 gates \d+", o) or [None])[0] for name, o in out.items()}
checks = {
    "seed still scores 89.0% / 28 gates": score["seed"] == EXPECT,
    "pinned start rejected": "started from the same angles" in out["pinned"] and score["pinned"] is None,
    "old-style default_rng(SEED) start rejected": "started from the same angles" in out["old_style"],
    "SEED = 7 in a block changes nothing": score["seed_rebound"] == EXPECT,
}
for what, ok in checks.items():
    print(("ok    " if ok else "FAIL  ") + what)
if not all(checks.values()):
    for name, o in out.items():
        print(f"\n--- {name} ---\n" + o[-1500:])
    sys.exit(1)
