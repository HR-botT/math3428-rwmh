"""Run the whole pipeline in order."""
import subprocess, sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

steps = [
    ("validation",       "validation.py"),
    ("simulation_study", "simulation_study.py"),
    ("baseline_comparison", "baseline_comparison.py"),
    ("budget_study",     "budget_study.py"),
    ("make_figures",     "make_figures.py"),
]

if __name__ == "__main__":
    t0 = time.time()
    for name, script in steps:
        print(f"\n=== Running {name} ({script}) ===")
        subprocess.run([PY, os.path.join(HERE, script)], check=True)
    print(f"\nAll steps finished in {time.time()-t0:.1f}s")