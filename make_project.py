from pathlib import Path
import json

TREE = [
    "clp/",

    "clp/README.md",
    "clp/pyproject.toml",
    "clp/.gitignore",

    "clp/clp/",
    "clp/clp/__init__.py",
    "clp/clp/config.py",

    "clp/clp/core/",
    "clp/clp/core/geometry.py",
    "clp/clp/core/cg.py",
    "clp/clp/core/ulo.py",
    "clp/clp/core/placement_points.py",
    "clp/clp/core/feasibility.py",

    "clp/clp/decoders/",
    "clp/clp/decoders/__init__.py",
    "clp/clp/decoders/random_ep.py",
    "clp/clp/decoders/first_feasible.py",
    "clp/clp/decoders/dblf.py",
    "clp/clp/decoders/dblf_balance.py",

    "clp/clp/ga/",
    "clp/clp/ga/__init__.py",
    "clp/clp/ga/nsga2.py",
    "clp/clp/ga/population.py",
    "clp/clp/ga/operators.py",
    "clp/clp/ga/fitness.py",

    "clp/clp/data/",
    "clp/clp/data/models.py",
    "clp/clp/data/io.py",
    "clp/clp/data/br_modify.py",

    "clp/clp/experiments/",
    "clp/clp/experiments/__init__.py",
    "clp/clp/experiments/variants.py",
    "clp/clp/experiments/runners.py",
    "clp/clp/experiments/baselines.py",
    "clp/clp/experiments/triobjective.py",
    "clp/clp/experiments/aggregation.py",

    "clp/clp/results/",
    "clp/clp/results/__init__.py",
    "clp/clp/results/schema.py",
    "clp/clp/results/writer.py",
    "clp/clp/results/index.py",

    "clp/clp/viz/",
    "clp/clp/viz/__init__.py",
    "clp/clp/viz/plot_pareto_2d.py",
    "clp/clp/viz/plot_pareto_3d.py",
    "clp/clp/viz/render_layout.py",

    "clp/datasets/",
    "clp/datasets/README.md",
    "clp/datasets/br_original/",
    "clp/datasets/br_modified/",
    "clp/datasets/br_weighted/",
    "clp/datasets/br_original/.keep",
    "clp/datasets/br_modified/.keep",
    "clp/datasets/br_weighted/.keep",

    "clp/results/",
    "clp/results/README.md",
    "clp/results/run_index.json",
    "clp/results/aggregated/",
    "clp/results/BR-Original/",
    "clp/results/BR-Modified/",
    "clp/results/aggregated/.keep",
    "clp/results/BR-Original/.keep",
    "clp/results/BR-Modified/.keep",

    "clp/scripts/",
    "clp/scripts/make_br_modified.py",
    "clp/scripts/run_experiments.py",
    "clp/scripts/aggregate_results.py",
]

GITIGNORE = """\
# Python
__pycache__/
*.pyc
.venv/
.env

# Outputs
results/*
!results/README.md
!results/run_index.json

# OS/editor
.DS_Store
.vscode/
"""

RESULTS_README = """\
This folder stores generated experiment outputs.
Do not commit large outputs to git.
"""

DATASETS_README = """\
Datasets:
- br_original/: raw BR instances
- br_modified/: generated tri-objective instances (seeded protocol)
- br_weighted/: optional weighted variants
"""

def ensure_file(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.is_dir():
        raise RuntimeError(f"Expected a file but found a directory: {path}")
    if not path.exists():
        path.write_text("", encoding="utf-8")

def ensure_dir(path: Path):
    if path.exists() and path.is_file():
        raise RuntimeError(f"Expected a directory but found a file: {path}")
    path.mkdir(parents=True, exist_ok=True)

def main():
    root = Path(".").resolve()

    # Create tree
    for rel in TREE:
        p = root / rel
        if rel.endswith("/"):
            ensure_dir(p)
        else:
            ensure_file(p)

    # Write key files safely
    gitignore = root / "clp/.gitignore"
    if gitignore.exists() and gitignore.is_dir():
        raise RuntimeError("clp/.gitignore exists as a directory. Delete/rename it first.")
    gitignore.write_text(GITIGNORE, encoding="utf-8")

    (root / "clp/results/README.md").write_text(RESULTS_README, encoding="utf-8")
    (root / "clp/datasets/README.md").write_text(DATASETS_README, encoding="utf-8")

    # Ensure run_index.json is valid JSON
    run_index = root / "clp/results/run_index.json"
    try:
        json.loads(run_index.read_text(encoding="utf-8") or "[]")
    except Exception:
        run_index.write_text("[]\n", encoding="utf-8")

    print("✅ Project scaffold created under ./clp/")

if __name__ == "__main__":
    main()
