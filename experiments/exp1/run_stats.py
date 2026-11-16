"""Per-generation stats for one Moses-branch run (run on PACE, writes JSON).

    python run_stats.py <run_dir> <models_dir> <results_dir> <out.json> [<gen> <cutoff_epoch> ...]

<run_dir>/<gen>/ holds one <gene>.sh per circuit evaluated in that generation (gen 0 is
the starting population). A circuit counts as valid when it wrote <gene>_metrics.json.
For each circuit we also record:
  fixed_start    its training start is a hard-coded number (all 10 trainings identical),
                 including SEED = <number> or default_rng(SEED) in the training block (since
                 2026-11-16 SEED stays 42 and main() hands train_angles its start as w0)
  unseeded_start its training block draws its own random start instead of using w0
                 (fair, not reproducible)
  readout_trick  it rewrote the readout (a softmax/exp, or N_CLASSES set to something but 3)
  seed_clone     it scores exactly like the seed (89.0% val, 64.0% soft, 28 gates)
  duplicate      QD_DEDUP skipped it: it behaves like a circuit already scored (not valid)
Valid circuits also get seconds (training time, all 10 trainings), val_sd (spread of the 10
validation scores; 0 means one training repeated) and depth. submitted / scored are the epoch
times its job script and its metrics file were written, for wall-clock time per generation.
Each optional <gen> <cutoff_epoch> pair drops that generation's circuits submitted before the
cutoff: children of an attempt that was cut off by the 8-hour limit and redone on resume.
"""
import glob, json, os, re, sys

FIXED_RE = re.compile(r"default_rng\(\s*\d+\s*\)|np\.random\.seed\(\s*\d+\s*\)|random_state\s*=\s*\d+")
LOCAL_SEED_RE = re.compile(r"^\s+SEED\s*=\s*\d+", re.M)  # pins SEED inside train_angles; default_rng(SEED) then reads it
OWN_RNG_RE = re.compile(r"default_rng\(|np\.random\.|\brandom\.")  # the block draws random numbers itself

run_dir, models, results, out = sys.argv[1:5]
cutoffs = {int(g): float(t) for g, t in zip(sys.argv[5::2], sys.argv[6::2])}  # gen -> epoch
MARK = "# --OPTION--"


def readout_trick(evolved):
    m = re.search(r"def readout_probabilities.*?(?=\ndef |\Z)", evolved, re.S)
    n = re.search(r"^\s*N_CLASSES\s*=\s*(\d+)", evolved, re.M)
    return bool(m and re.search(r"exp\(|softmax", m.group(0))) or bool(n and int(n.group(1)) != 3)


rows, dropped = [], 0
for gdir in sorted(glob.glob(os.path.join(run_dir, "[0-9]*")), key=lambda p: int(os.path.basename(p))):
    gen = int(os.path.basename(gdir))
    for sh in sorted(glob.glob(os.path.join(gdir, "*.sh"))):
        gene = os.path.basename(sh)[:-3]
        if gene.endswith("_model"):
            continue
        if gen in cutoffs and os.path.getmtime(sh) < cutoffs[gen]:
            dropped += 1
            continue
        src_path = os.path.join(models, f"network_{gene}.py")
        src = open(src_path, errors="ignore").read().replace("\r", "") if os.path.exists(src_path) else ""
        evolved = MARK.join(src.split(MARK)[1:])
        train = src.split(MARK)[-1]
        r = {"gen": gen, "gene": gene, "made": bool(src), "submitted": os.path.getmtime(sh),
             "fixed_start": bool(src) and bool(LOCAL_SEED_RE.search(train) or FIXED_RE.search(train)
                                               or "default_rng(SEED)" in train),
             "unseeded_start": bool(src) and bool(OWN_RNG_RE.search(train)) and not FIXED_RE.search(train)
                               and "default_rng(SEED)" not in train,
             "readout_trick": bool(src) and readout_trick(evolved)}
        mpath = os.path.join(results, f"{gene}_metrics.json")
        m = json.load(open(mpath)) if os.path.exists(mpath) else {}
        r["duplicate"] = bool(m.get("duplicate"))  # QD_DEDUP: behaves like a circuit already scored, not trained
        if m and not r["duplicate"]:
            r.update(valid=True, val=m["val_accuracy"], soft=m.get("val_soft_accuracy"),
                     gates=m["gates"], cnots=m["two_qubit_gates"], test=m.get("test_accuracy"),
                     seconds=m.get("seconds"), val_sd=m.get("val_accuracy_sd"), depth=m.get("depth"), n_qubits=m.get("n_qubits"),
                     scored=os.path.getmtime(mpath))
            r["seed_clone"] = (abs(r["val"] - 0.89) < 1e-6 and r["gates"] == 28
                               and r["soft"] is not None and abs(r["soft"] - 0.6396) < 5e-4)
        else:
            r["valid"] = False
        rows.append(r)
json.dump(rows, open(out, "w"))
print(f"{len(rows)} circuits over {1 + max(r['gen'] for r in rows)} generations -> {out}"
      + (f" ({dropped} circuits from cut-off attempts left out)" if dropped else ""))
