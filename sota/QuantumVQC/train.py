"""
Adapter so the quantum seed satisfies LLM-GE's eval contract.

run_improved.py invokes every individual as EVAL_RUNLINE in src/cfg/constants.py:

    uv run python <TRAIN_FILE> --model network_<gene_id> --variant_dir <VARIANT_DIR>

the same signature sota/Titanic/eval.py takes. Our seed is self-contained - the
mutated file builds its circuit, trains the angles, scores on validation and writes
its own results file - so this shim only imports the variant and calls its main().
"""

import argparse
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="network", help='e.g. "network_abc123"')
    ap.add_argument("--variant_dir", default=os.path.join(HERE, "models"))
    args, unknown = ap.parse_known_args()
    if unknown:
        print(f"[train.py] ignoring unused args: {unknown}")

    # "network_<gene_id>" -> "<gene_id>"; the bare seed is "network" -> "seed"
    gene_id = args.model.split("network_", 1)[1] if "network_" in args.model else "seed"

    # The seed itself lives next to this file, variants in variant_dir.
    sys.path[:0] = [args.variant_dir, HERE]
    module = importlib.import_module(args.model)

    # check4results() reads SOTA_ROOT/results/<gene_id>_results.csv.
    sys.argv = ["network", "--gene-id", gene_id, "--out-dir", os.path.join(HERE, "results")]
    module.main()

    # check_contents_for_error() treats a job as finished only when this exact
    # string appears in its slurm log. Without it the orchestrator waits out its
    # whole walltime on a job that finished in 20 seconds.
    print("=" * 120); print("job done"); print("=" * 120)


if __name__ == "__main__":
    main()
