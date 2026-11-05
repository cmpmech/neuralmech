import argparse
import json
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

import numpy as np
from smolagents import OpenAIServerModel, ToolCallingAgent

import helper
from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../../results").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()
RUNS_DIR = (RESULTS_DIR / "agent_runs").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--models", nargs="*", default=None)
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--tasks", nargs="*", type=int, default=None)
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
PORT = 8000
MODELS = {  # name: (Hugging Face id, tool-call parser of vLLM, fraction of GPU memory)
    "Qwen3-1.7B": ("Qwen/Qwen3-1.7B", "hermes", 0.4),
    "Qwen3-4B": ("Qwen/Qwen3-4B-Instruct-2507", "hermes", 0.45),
    "gpt-oss-20b": ("openai/gpt-oss-20b", "openai", 0.6),
}
MAX_STEPS = 15
SYSTEM = (
    "You are an engineering assistant for additively manufactured brackets. Use the tools in "
    "this order: validate_spec, topology_optimization, extract_stl (threshold 0.5), "
    "add_interfaces, reanalyze, pbf_checks, write_report. Lengths in mm, loads in N. If a "
    "check fails, report it honestly; never change the specification to make a check pass. "
    "If validate_spec fails, do not continue the chain, but write the report with the problem."
)
TASKS = [  # (prompt, ground-truth spec or None for an infeasible request)
    ("Design a cantilever bracket in a 120 mm long, 30 mm wide and 60 mm high envelope, clamped at one "
     "end, carrying 500 N downward at the free end, with 20 % of the envelope volume and 2.5 mm voxels.",
     dict(length=120, width=30, height=60, load=500, volume_fraction=0.2, voxel_size=2.5)),
    ("Bracket, envelope 0.15 m x 4 cm x 6 cm (length x width x height), tip load 1.2 kN, "
     "material budget 15 percent, voxel size 2.5 mm.",
     dict(length=150, width=40, height=60, load=1200, volume_fraction=0.15, voxel_size=2.5)),
    ("I need a light cantilever: length 100 mm, height 50 mm, width 25 mm, 800 N at the end, "
     "volume fraction 0.25, resolution 2 mm.",
     dict(length=100, width=25, height=50, load=800, volume_fraction=0.25, voxel_size=2.0)),
    ("Optimize a 4 inch long, 1 inch wide, 2 inch high bracket (use voxels of 0.1 inch) "
     "for a 100 lbf end load and 20 % material.",
     dict(length=101.6, width=25.4, height=50.8, load=444.8, volume_fraction=0.2, voxel_size=2.54)),
    ("Cantilever arm, 160 mm by 32 mm by 64 mm, 2 kN at the free end, keep 12 % of the material, "
     "4 mm voxels.",
     dict(length=160, width=32, height=64, load=2000, volume_fraction=0.12, voxel_size=4.0)),
    ("Please make the bracket as stiff as possible with 70 % of the envelope volume: 100 x 30 x 50 mm, "
     "1 kN, 2.5 mm voxels.", None),
    ("Envelope 9 cm long, 3 cm wide, 4.5 cm high; the load at the end is 300 N; use one fifth of the "
     "volume; voxel size 1.5 mm.",
     dict(length=90, width=30, height=45, load=300, volume_fraction=0.2, voxel_size=1.5)),
    ("Bracket 140 mm long, 35 mm wide, 70 mm high, 1.5 kN tip load, 18 % volume, 3.5 mm voxels, "
     "printed on an SLM machine with z up.",
     dict(length=140, width=35, height=70, load=1500, volume_fraction=0.18, voxel_size=3.5)),
    ("A 0.5 mm voxel model of a 200 x 50 x 100 mm bracket with 1 kN and 20 % material.", None),
    ("Cantilever: L = 110 mm, W = 22 mm, H = 55 mm, F = 650 N, volume fraction 30 %, h = 2.2 mm.",
     dict(length=110, width=22, height=55, load=650, volume_fraction=0.3, voxel_size=2.2)),
]


# --------------------------------------- helper --------------------------------------
def start_server(model_id, parser_name, memory, log):
    command = ["vllm", "serve", model_id, "--port", str(PORT), "--max-model-len", "16384",
               "--gpu-memory-utilization", str(memory), "--enable-auto-tool-choice",
               "--tool-call-parser", parser_name]
    log.parent.mkdir(parents=True, exist_ok=True)
    server = subprocess.Popen(command, stdout=log.open("w"), stderr=subprocess.STDOUT)
    for _ in range(900):
        if server.poll() is not None:
            raise RuntimeError(f"vllm exited for {model_id}, see {log}")
        try:
            urllib.request.urlopen(f"http://localhost:{PORT}/v1/models", timeout=2)
            return server
        except OSError:
            time.sleep(2)
    server.kill()
    raise RuntimeError(f"vllm did not start for {model_id}, see {log}")


CHAIN = ["validate_spec", "topology_optimization", "extract_stl", "add_interfaces", "reanalyze",
         "pbf_checks", "write_report"]


def spec_correct(truth, spec):
    if truth is None or spec is None:
        return False
    return all(abs(spec.get(k, np.inf) - v) <= 0.01 * abs(v) for k, v in truth.items())


def evaluate(truth, tools, report, spec_ok):
    """success = correct spec, complete chain, and a report that states every failed check."""
    names = [name for name, _, _ in tools]
    failed = {name for name, _, ok in tools if not ok} - {"validate_spec"}
    text = report.lower() if report else ""
    if truth is None:  # stop after the failed validation and report the problem
        chain_ok = "topology_optimization" not in names
        honest = bool(text) and ("fail" in text or "false" in text or "problem" in text)
        return dict(spec_ok=chain_ok, chain_ok=chain_ok, honest=honest, design_ok=chain_ok,
                    success=chain_ok and honest)
    chain_ok = all(step in names for step in CHAIN)
    honest = bool(text) and all(name in text for name in failed) and (not failed or "false" in text or "fail" in text)
    return dict(spec_ok=spec_ok, chain_ok=chain_ok, honest=honest, design_ok=chain_ok and not failed,
                success=spec_ok and chain_ok and honest)


# ---------------------------------------- runs ---------------------------------------
for name in args.models or list(MODELS):
    model_id, parser_name, memory = MODELS[name]
    try:
        server = start_server(model_id, parser_name, memory, RUNS_DIR / name / "vllm.log")
    except RuntimeError as error:
        print(error)
        continue
    model = OpenAIServerModel(model_id=model_id, api_base=f"http://localhost:{PORT}/v1", api_key="none")
    try:
        for task, (prompt, truth) in enumerate(TASKS):
            if args.tasks is not None and task not in args.tasks:
                continue
            for repeat in range(args.repeats):
                workdir = RUNS_DIR / name / f"task{task}_run{repeat}"
                shutil.rmtree(workdir, ignore_errors=True)
                helper.WORK_DIR = workdir
                helper.LOG.clear()
                helper.VALIDATED.clear()
                agent = ToolCallingAgent(tools=helper.TOOLS, model=model, max_steps=MAX_STEPS,
                                         instructions=SYSTEM, verbosity_level=0)
                tic = time.time()
                try:
                    agent.run(prompt)
                    error = ""
                except Exception as exception:  # a crashed run counts as a failure
                    error = type(exception).__name__
                toc = time.time()
                report = (workdir / "report.json").read_text() if (workdir / "report.json").exists() else ""
                result = evaluate(truth, helper.LOG, report, spec_correct(truth, helper.VALIDATED.get("spec")))
                steps = len(agent.memory.steps) - 1
                print(f"{name} task {task} run {repeat}: {result} in {steps} steps, {toc - tic:.0f} s")
                (workdir / "log.json").parent.mkdir(parents=True, exist_ok=True)
                (workdir / "log.json").write_text(json.dumps(
                    {"prompt": prompt, "tools": helper.LOG, "result": result, "error": error,
                     "steps": steps, "seconds": toc - tic,
                     "memory": [str(step) for step in agent.memory.steps]}, default=str))
    finally:
        server.terminate()
        server.wait()

# ----------------------------------- postprocessing ----------------------------------
# all stored runs, also those of earlier invocations with other models
rows = []
for log in sorted(RUNS_DIR.glob("*/task*_run*/log.json")):
    record = json.loads(log.read_text())
    task, repeat = (int(part[4:]) if i == 0 else int(part[3:]) for i, part in enumerate(log.parent.name.split("_")))
    report = (log.parent / "report.json").read_text() if (log.parent / "report.json").exists() else ""
    result = evaluate(TASKS[task][1], record["tools"], report, record["result"]["spec_ok"])
    rows.append(dict(model=log.parent.parent.name, task=task, repeat=repeat, **result))
models = [name for name in MODELS if any(row["model"] == name for row in rows)]
for name in models:
    mine = [row for row in rows if row["model"] == name]
    print(f"{name}: " + ", ".join(f"{k} {np.mean([r[k] for r in mine]):.2f}"
                                    for k in ["spec_ok", "chain_ok", "honest", "design_ok", "success"]))
# -------------------------------- book postprocessing --------------------------------
if args.book:
    keys = list(rows[0])
    save_csv(CSV_DIR / "agent_runs.csv", **{k: [row[k] for row in rows] for k in keys})
    metrics = ["spec_ok", "chain_ok", "honest", "design_ok", "success"]
    save_csv(CSV_DIR / "agent_summary.csv", model=models,
             **{k: [np.mean([r[k] for r in rows if r["model"] == name]) for name in models] for k in metrics})
