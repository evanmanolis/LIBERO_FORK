#!/usr/bin/env python3
"""
Kaggle-friendly LIBERO runner.

Purpose
-------
This script is designed to be uploaded to a Kaggle notebook and run as a single,
clean entrypoint for:
  1) environment setup,
  2) periodic artifact snapshotting,
  3) task execution,
  4) packaging results.

It keeps the original workflow conceptually the same, while reducing a few avoidable
sources of overhead and instability:
  - snapshot copies happen periodically, but zipping happens only once at the end;
  - the virtual environment is not rebuilt unless requested;
  - subprocess logging is centralized;
  - configuration is grouped into a dataclass;
  - a self-test mode is included for non-Kaggle validation.

Notes
-----
- This script cannot fully validate a Kaggle GPU/MuJoCo/LIBERO stack outside Kaggle.
- The `--self-test` mode validates the internal control flow, packaging, snapshotting,
  and fake task execution without requiring Kaggle, a GPU, or the LIBERO repo.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import pathlib
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import textwrap
import threading
import time
import zipfile
from dataclasses import dataclass, field, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Set


# Optional dependency: only required for real setup mode.
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - allowed in self-test when absent
    yaml = None


# -----------------------------
# Configuration
# -----------------------------

@dataclass
class RunnerConfig:
    # Paths
    repo_url: str = "https://github.com/Lifelong-Robot-Learning/LIBERO.git"
    repo_dir: str = "/kaggle/working/LIBERO"
    venv_dir: str = "/kaggle/working/libero_env"
    libero_config_path: str = "/kaggle/working/.libero"
    datasets_root: str = "/kaggle/working/libero_datasets"
    runs_root: str = "/kaggle/working/libero_runs"
    meta_path: str = "/kaggle/working/libero_setup_meta.json"

    # Snapshots
    src_root: str = "/kaggle/working/LIBERO/experiments"
    snap_root: str = "/kaggle/working/libero_snapshots"
    zip_path: str = "/kaggle/working/libero_snapshots.zip"
    keep_exts: Set[str] = field(default_factory=lambda: {".pth", ".pt", ".log", ".json", ".txt"})
    snapshot_period_sec: int = 300

    # Setup behavior
    install_os_packages: bool = True
    rebuild_env: bool = False
    download_datasets: bool = True

    # Training config
    run_stage: str = "full"  # canary0 | canary1 | canary2 | full
    seed: int = 10000
    benchmark: str = "LIBERO_OBJECT"
    policy: str = "bc_transformer_policy"
    algo: str = "base"
    skip_flops: bool = True
    eval_num_procs: int = 1
    target_task_ids: List[int] = field(default_factory=lambda: [3, 4])
    package_name: str = "libero_object_tasks_3_4_results"
    extra_overrides: List[str] = field(default_factory=list)

    # Timeouts
    setup_timeout_min: int = 120
    full_timeout_min: int = 720

    # Misc.
    torch_install_spec: str = (
        "torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 "
        "--index-url https://download.pytorch.org/whl/cu118"
    )


# -----------------------------
# Subprocess helpers
# -----------------------------

class CommandError(RuntimeError):
    pass


def print_header(title: str) -> None:
    line = "=" * 80
    print(f"\n{line}\n{title}\n{line}\n", flush=True)


def kill_process_group(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            proc.kill()


def run_stream(
    cmd: str,
    *,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None,
    timeout_min: Optional[int] = None,
    heartbeat_s: int = 30,
    check: bool = True,
) -> int:
    """Run a shell command with streamed output and periodic heartbeats."""
    print(f"\n$ {cmd}\n", flush=True)
    start = time.time()
    last_output = start
    proc = subprocess.Popen(
        cmd,
        shell=True,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid,
    )
    assert proc.stdout is not None

    try:
        while True:
            now = time.time()
            if timeout_min is not None and (now - start) > timeout_min * 60:
                kill_process_group(proc)
                raise CommandError(f"Timeout after {timeout_min} min: {cmd}")

            line = proc.stdout.readline()
            if line:
                print(line, end="", flush=True)
                last_output = time.time()
                continue

            if proc.poll() is not None:
                remainder = proc.stdout.read()
                if remainder:
                    print(remainder, end="", flush=True)
                break

            if (now - last_output) >= heartbeat_s:
                elapsed = (now - start) / 60.0
                print(f"[heartbeat] running | elapsed={elapsed:.1f} min", flush=True)
                last_output = now

        rc = int(proc.returncode or 0)
        if check and rc != 0:
            raise CommandError(f"Command failed ({rc}): {cmd}")
        return rc
    finally:
        kill_process_group(proc)


# -----------------------------
# Snapshot manager
# -----------------------------

class SnapshotManager:
    def __init__(self, src_root: str, snap_root: str, zip_path: str, keep_exts: Set[str], period_sec: int) -> None:
        self.src_root = src_root
        self.snap_root = snap_root
        self.zip_path = zip_path
        self.keep_exts = {ext.lower() for ext in keep_exts}
        self.period_sec = period_sec
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        pathlib.Path(self.snap_root).mkdir(parents=True, exist_ok=True)

    def snapshot_once(self) -> int:
        if not os.path.exists(self.src_root):
            return 0
        copied = 0
        for root, _, files in os.walk(self.src_root):
            for name in files:
                if pathlib.Path(name).suffix.lower() not in self.keep_exts:
                    continue
                src = os.path.join(root, name)
                rel = os.path.relpath(src, self.src_root)
                dst = os.path.join(self.snap_root, rel)
                pathlib.Path(os.path.dirname(dst)).mkdir(parents=True, exist_ok=True)
                if (not os.path.exists(dst)) or os.path.getmtime(src) > os.path.getmtime(dst):
                    shutil.copy2(src, dst)
                    copied += 1
        print(f"[snapshot] copied/updated {copied} files", flush=True)
        return copied

    def zip_snapshots(self) -> str:
        with zipfile.ZipFile(self.zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(self.snap_root):
                for name in files:
                    full = os.path.join(root, name)
                    rel = os.path.relpath(full, self.snap_root)
                    zf.write(full, rel)
        print(f"[snapshot] wrote zip: {self.zip_path}", flush=True)
        return self.zip_path

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                self.snapshot_once()
            except Exception as exc:
                print(f"[snapshot] error: {exc}", flush=True)
            self.stop_event.wait(self.period_sec)

    def start(self) -> None:
        if self.thread is not None:
            return
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        print(f"Snapshot thread started: {self.snap_root}", flush=True)

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2)


# -----------------------------
# Setup helpers
# -----------------------------

def detect_python() -> str:
    for candidate in (shutil.which("python3.10"), shutil.which("python3"), sys.executable):
        if candidate:
            return candidate
    raise RuntimeError("No Python executable found")


def kaggle_env(base_env: Dict[str, str], repo_dir: str, libero_config_path: str, *, limit_threads: bool) -> Dict[str, str]:
    env = dict(base_env)
    env["MUJOCO_GL"] = "egl"
    env["PYOPENGL_PLATFORM"] = "egl"
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
    env["MUJOCO_EGL_DEVICE_ID"] = env.get("MUJOCO_EGL_DEVICE_ID", env["CUDA_VISIBLE_DEVICES"])
    env["WANDB_MODE"] = "offline"
    env["HYDRA_FULL_ERROR"] = "1"
    env["LIBERO_CONFIG_PATH"] = libero_config_path
    env["PYTHONPATH"] = f"{repo_dir}:" + env.get("PYTHONPATH", "")
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONUNBUFFERED"] = "1"
    if limit_threads:
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["NUMEXPR_NUM_THREADS"] = "1"
    return env


def write_robosuite_macro_stub(path: str) -> None:
    pathlib.Path(path).write_text("# autogenerated\n", encoding="utf-8")


def gpu_diagnostics() -> None:
    print_header("GPU DIAGNOSTICS")
    try:
        import torch  # type: ignore

        print("torch:", torch.__version__)
        print("cuda available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("gpu:", torch.cuda.get_device_name(0))
            print("capability:", torch.cuda.get_device_capability(0))
    except Exception as exc:
        print(f"torch import failed: {exc}")

    try:
        print("\n=== nvidia-smi ===")
        print(subprocess.check_output("nvidia-smi", shell=True, text=True))
    except Exception as exc:
        print(f"nvidia-smi unavailable: {exc}")


def validate_network_hosts(hosts: Sequence[str]) -> None:
    print_header("NETWORK CHECK")
    for host in hosts:
        try:
            print(host, "->", socket.gethostbyname(host))
        except Exception as exc:
            print(host, "-> resolution failed:", exc)


def ensure_repo(cfg: RunnerConfig) -> None:
    if not os.path.exists(cfg.repo_dir):
        run_stream(f"git clone {shlex.quote(cfg.repo_url)} {shlex.quote(cfg.repo_dir)}", timeout_min=10)
    else:
        print(f"Repo exists: {cfg.repo_dir}")


def ensure_venv(cfg: RunnerConfig, py_exec: str) -> str:
    if cfg.rebuild_env and os.path.exists(cfg.venv_dir):
        shutil.rmtree(cfg.venv_dir)

    py_bin = os.path.join(cfg.venv_dir, "bin", "python")
    if not os.path.exists(py_bin):
        rc = run_stream(f"{py_exec} -m venv {shlex.quote(cfg.venv_dir)}", check=False, timeout_min=5)
        if rc != 0:
            run_stream(f"{py_exec} -m pip install -U virtualenv", timeout_min=5)
            run_stream(f"{py_exec} -m virtualenv {shlex.quote(cfg.venv_dir)}", timeout_min=5)
    return py_bin


def install_runtime(cfg: RunnerConfig, py_bin: str) -> None:
    run_stream(f"{py_bin} -m pip install -U pip setuptools==65.5.1 wheel packaging", timeout_min=10)
    run_stream(f"{py_bin} -m pip install --no-cache-dir {cfg.torch_install_spec}", timeout_min=30)
    run_stream(
        f'{py_bin} -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"',
        timeout_min=3,
    )


def patch_requirements(repo_dir: str) -> str:
    req_src = pathlib.Path(repo_dir) / "requirements.txt"
    req_tmp = pathlib.Path("/kaggle/working/requirements.kaggle.txt")
    patched: List[str] = []
    for line in req_src.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("numpy=="):
            patched.append("numpy==1.23.5")
        elif s.startswith("torch==") or s.startswith("torchvision==") or s.startswith("torchaudio=="):
            continue
        else:
            patched.append(line)
    req_tmp.write_text("\n".join(patched) + "\n", encoding="utf-8")
    return str(req_tmp)


def write_libero_config(cfg: RunnerConfig) -> str:
    if yaml is None:
        raise RuntimeError("PyYAML is required for setup mode")

    pathlib.Path(cfg.libero_config_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(cfg.datasets_root).mkdir(parents=True, exist_ok=True)
    pathlib.Path(cfg.runs_root).mkdir(parents=True, exist_ok=True)

    benchmark_root = f"{cfg.repo_dir}/libero/libero"
    out_path = f"{cfg.libero_config_path}/config.yaml"
    payload = {
        "benchmark_root": benchmark_root,
        "bddl_files": f"{benchmark_root}/bddl_files",
        "init_states": f"{benchmark_root}/init_files",
        "datasets": cfg.datasets_root,
        "assets": f"{benchmark_root}/assets",
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh)
    return benchmark_root


def maybe_install_os_packages() -> None:
    run_stream("apt-get update -y", timeout_min=15)
    run_stream(
        "DEBIAN_FRONTEND=noninteractive apt-get install -y "
        "python3.10 python3.10-venv python3.10-distutils "
        "libosmesa6-dev libgl1-mesa-glx libglfw3 libglew-dev patchelf ffmpeg",
        timeout_min=25,
    )


def maybe_download_datasets(cfg: RunnerConfig, py_bin: str, env: Dict[str, str]) -> None:
    existing = glob.glob(f"{cfg.datasets_root}/libero_object/*_demo.hdf5")
    if len(existing) >= 10 or not cfg.download_datasets:
        return
    run_stream(
        f"{py_bin} benchmark_scripts/download_libero_datasets.py --datasets libero_object --use-huggingface",
        env=env,
        cwd=cfg.repo_dir,
        timeout_min=40,
    )

    files = sorted(glob.glob(f"{cfg.datasets_root}/libero_object/*_demo.hdf5"))
    print(f"libero_object demos: {len(files)}")
    for path in files:
        print(" -", os.path.basename(path))
    if len(files) != 10:
        raise RuntimeError(f"Expected 10 demo files, found {len(files)}")


def write_setup_meta(cfg: RunnerConfig, py_bin: str, benchmark_root: str) -> None:
    payload = {
        "PY": py_bin,
        "REPO_DIR": cfg.repo_dir,
        "LIBERO_CONFIG_PATH": cfg.libero_config_path,
        "DATASETS_ROOT": cfg.datasets_root,
        "RUNS_ROOT": cfg.runs_root,
        "benchmark_root": benchmark_root,
        "config": asdict(cfg),
    }
    pathlib.Path(cfg.meta_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Setup complete. Meta: {cfg.meta_path}")


def setup_environment(cfg: RunnerConfig) -> None:
    print_header("SETUP")
    gpu_diagnostics()
    validate_network_hosts(["pypi.org", "github.com", "huggingface.co"])

    if cfg.install_os_packages:
        maybe_install_os_packages()

    py_exec = detect_python()
    print("Host Python:", sys.version)
    run_stream("nvidia-smi -L", timeout_min=2, check=False)

    ensure_repo(cfg)
    py_bin = ensure_venv(cfg, py_exec)
    install_runtime(cfg, py_bin)

    req_path = patch_requirements(cfg.repo_dir)
    print(f"Using requirements: {req_path}")
    run_stream(f"{py_bin} -m pip install --prefer-binary -r {req_path}", cwd=cfg.repo_dir, timeout_min=45)
    run_stream(f"{py_bin} -m pip install -e .", cwd=cfg.repo_dir, timeout_min=12)

    benchmark_root = write_libero_config(cfg)
    env = kaggle_env(os.environ, cfg.repo_dir, cfg.libero_config_path, limit_threads=False)

    macro_path = "/kaggle/working/robosuite_macros_private.py"
    write_robosuite_macro_stub(macro_path)
    env["ROBOSUITE_MACROS_PRIVATE_PATH"] = macro_path
    env["NUMBA_DISABLE_JIT"] = "1"

    run_stream(
        f"{py_bin} -c \"import hydra,yaml,imageio,h5py,robomimic,robosuite,torch; "
        f"print('imports OK | torch', torch.__version__)\"",
        env=env,
        cwd=cfg.repo_dir,
        timeout_min=5,
    )

    maybe_download_datasets(cfg, py_bin, env)
    write_setup_meta(cfg, py_bin, benchmark_root)


# -----------------------------
# Run helpers
# -----------------------------

def stage_overrides_and_timeout(run_stage: str, full_timeout_min: int) -> tuple[list[str], int]:
    if run_stage == "canary0":
        return ([
            "train.n_epochs=0", "eval.eval=false",
            "eval.eval_every=999999", "eval.n_eval=1", "eval.max_steps=1",
        ], 30)
    if run_stage == "canary1":
        return ([
            "train.n_epochs=1", "eval.eval=false",
            "eval.eval_every=999999", "eval.n_eval=1", "eval.max_steps=1",
        ], 75)
    if run_stage == "canary2":
        return ([
            "train.n_epochs=1", "eval.eval=true",
            "eval.eval_every=1", "eval.n_eval=1", "eval.max_steps=50",
        ], 300)
    if run_stage == "full":
        return ([], full_timeout_min)
    raise ValueError("run_stage must be one of: canary0, canary1, canary2, full")


def load_setup_meta(meta_path: str) -> dict:
    if not os.path.exists(meta_path):
        raise RuntimeError(f"Setup meta not found: {meta_path}. Run setup first.")
    return json.loads(pathlib.Path(meta_path).read_text(encoding="utf-8"))


def write_launcher(
    *,
    target_task_id: int,
    py_path: str,
    repo_dir: str,
    benchmark_root: str,
    datasets_root: str,
    run_dir: str,
    cfg: RunnerConfig,
    stage_overrides: Sequence[str],
) -> str:
    overrides = [
        f"seed={cfg.seed}",
        f"benchmark_name={cfg.benchmark}",
        f"policy={cfg.policy}",
        f"lifelong={cfg.algo}",
        f"folder={datasets_root}",
        f"bddl_folder={benchmark_root}/bddl_files",
        f"init_states_folder={benchmark_root}/init_files",
        "device=cuda:0",
        "use_wandb=false",
        "train.num_workers=0",
        "eval.num_workers=0",
        f"hydra.run.dir={run_dir}",
        "hydra.job.chdir=false",
        "eval.use_mp=true",
        f"eval.num_procs={cfg.eval_num_procs}",
    ] + list(stage_overrides) + list(cfg.extra_overrides)

    launcher_code = textwrap.dedent(f'''
        import multiprocessing as mp
        import sys

        if mp.get_start_method(allow_none=True) != "fork":
            mp.set_start_method("fork", force=True)
        try:
            import torch.multiprocessing as tmp
            tmp.set_start_method("fork", force=True)
        except Exception:
            pass

        import torch
        print("[launcher] torch:", torch.__version__, "| cuda:", torch.version.cuda, flush=True)
        print("[launcher] cuda available:", torch.cuda.is_available(), flush=True)
        if torch.cuda.is_available():
            print("[launcher] gpu:", torch.cuda.get_device_name(0), flush=True)

        import torch.utils.data as tud
        _orig_init = tud.DataLoader.__init__
        def _patched_init(self, *args, **kwargs):
            if kwargs.get("num_workers", 0) == 0:
                kwargs["persistent_workers"] = False
            return _orig_init(self, *args, **kwargs)
        tud.DataLoader.__init__ = _patched_init

        import libero.libero.benchmark as bm
        _orig_get_benchmark = bm.get_benchmark

        def _task_only_get_benchmark(name):
            base_cls = _orig_get_benchmark(name)
            class OneTaskBenchmark(base_cls):
                def __init__(self, task_order_index=0):
                    super().__init__(task_order_index)
                    if {target_task_id} >= len(self.tasks):
                        raise IndexError("task id out of range")
                    self.tasks = [self.tasks[{target_task_id}]]
                    self.n_tasks = 1
            return OneTaskBenchmark

        bm.get_benchmark = _task_only_get_benchmark

        import libero.lifelong.main as lm
        lm.get_benchmark = _task_only_get_benchmark

        if {cfg.skip_flops!r}:
            def _skip_flops(*args, **kwargs):
                print("[launcher] skipping compute_flops", flush=True)
                return 0.0, 0.0
            lm.compute_flops = _skip_flops

        sys.argv = ["libero.lifelong.main"] + {overrides!r}
        print("[launcher] running task id {target_task_id}", flush=True)
        lm.main()
        print("[launcher] done", flush=True)
    ''').strip() + "\n"

    launcher_path = str(
        pathlib.Path(run_dir).parent / f"run_libero_task_{target_task_id}.py"
    )
    pathlib.Path(launcher_path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(launcher_path).write_text(launcher_code, encoding="utf-8")
    return launcher_path


def package_results(bundle_dir: pathlib.Path, zip_base: str) -> str:
    zip_path = pathlib.Path(zip_base + ".zip")
    if zip_path.exists():
        zip_path.unlink()
    shutil.make_archive(zip_base, "zip", bundle_dir)
    return str(zip_path)


def run_tasks(cfg: RunnerConfig) -> str:
    print_header("RUN TASKS")
    meta = load_setup_meta(cfg.meta_path)
    py_bin = meta["PY"]
    repo_dir = meta["REPO_DIR"]
    libero_config_path = meta["LIBERO_CONFIG_PATH"]
    datasets_root = meta["DATASETS_ROOT"]
    runs_root = meta["RUNS_ROOT"]
    benchmark_root = meta["benchmark_root"]

    env = kaggle_env(os.environ, repo_dir, libero_config_path, limit_threads=True)
    macro_path = "/kaggle/working/robosuite_macros_private.py"
    write_robosuite_macro_stub(macro_path)
    env["ROBOSUITE_MACROS_PRIVATE_PATH"] = macro_path

    stage_overrides, timeout_min = stage_overrides_and_timeout(cfg.run_stage, cfg.full_timeout_min)

    experiments_root = pathlib.Path(repo_dir) / "experiments" / cfg.benchmark / "Sequential" / f"BCTransformerPolicy_seed{cfg.seed}"
    experiments_root.mkdir(parents=True, exist_ok=True)

    bundle_dir = pathlib.Path(f"/kaggle/working/{cfg.package_name}")
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, object] = {"run_stage": cfg.run_stage, "tasks": [], "errors": []}

    for target_task_id in cfg.target_task_ids:
        print_header(f"Running task id {target_task_id}")
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = f"{runs_root}/{cfg.benchmark}/{cfg.policy}/{cfg.algo}/seed_{cfg.seed}/task{target_task_id}_{stamp}"
        pathlib.Path(run_dir).mkdir(parents=True, exist_ok=True)

        launcher_path = write_launcher(
            target_task_id=target_task_id,
            py_path=py_bin,
            repo_dir=repo_dir,
            benchmark_root=benchmark_root,
            datasets_root=datasets_root,
            run_dir=run_dir,
            cfg=cfg,
            stage_overrides=stage_overrides,
        )

        before_runs = {p.name for p in experiments_root.glob("run_*")}
        task_error: Optional[str] = None
        latest_run: Optional[pathlib.Path] = None

        try:
            run_stream(f"{py_bin} -u {launcher_path}", env=env, cwd=repo_dir, timeout_min=timeout_min)
        except Exception as exc:
            task_error = str(exc)
            cast_errors = summary["errors"]
            assert isinstance(cast_errors, list)
            cast_errors.append({"task_id": target_task_id, "error": task_error})
        finally:
            after_runs = sorted(
                [p for p in experiments_root.glob("run_*") if p.is_dir()],
                key=lambda p: p.stat().st_mtime,
            )
            for p in reversed(after_runs):
                if p.name not in before_runs:
                    latest_run = p
                    break
            if latest_run is None and after_runs:
                latest_run = after_runs[-1]

            task_dir = bundle_dir / f"task{target_task_id}"
            task_dir.mkdir(parents=True, exist_ok=True)

            manifest = {
                "task_id_zero_based": target_task_id,
                "benchmark": cfg.benchmark,
                "policy": cfg.policy,
                "algo": cfg.algo,
                "experiments_run_dir": str(latest_run) if latest_run else None,
                "launcher_run_dir": run_dir,
                "error": task_error,
            }
            (task_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

            if latest_run is not None:
                rename_map = {
                    "config.json": "config.json",
                    "task0_model.pth": f"task{target_task_id}_model.pth",
                    "task0_auc.log": f"task{target_task_id}_auc.log",
                    "result.pt": "result.pt",
                }
                for src_name, dst_name in rename_map.items():
                    src = latest_run / src_name
                    if src.exists():
                        shutil.copy2(src, task_dir / dst_name)

            cast_tasks = summary["tasks"]
            assert isinstance(cast_tasks, list)
            cast_tasks.append(manifest)

    (bundle_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    zip_path = package_results(bundle_dir, f"/kaggle/working/{cfg.package_name}")

    print(f"Packaged folder: {bundle_dir}")
    print(f"Packaged zip   : {zip_path}")

    errors = summary["errors"]
    assert isinstance(errors, list)
    if errors:
        raise RuntimeError(f"One or more tasks failed: {errors}")

    print(f"Done. Ran task IDs: {cfg.target_task_ids}")
    return zip_path


# -----------------------------
# Self-test mode
# -----------------------------

def create_fake_meta(base: str, cfg: RunnerConfig) -> dict:
    repo_dir = os.path.join(base, "LIBERO")
    exp_dir = os.path.join(repo_dir, "experiments", cfg.benchmark, "Sequential", f"BCTransformerPolicy_seed{cfg.seed}")
    pathlib.Path(exp_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(base, ".libero")).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(base, "datasets")).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(base, "runs")).mkdir(parents=True, exist_ok=True)

    meta = {
        "PY": sys.executable,
        "REPO_DIR": repo_dir,
        "LIBERO_CONFIG_PATH": os.path.join(base, ".libero"),
        "DATASETS_ROOT": os.path.join(base, "datasets"),
        "RUNS_ROOT": os.path.join(base, "runs"),
        "benchmark_root": os.path.join(repo_dir, "libero", "libero"),
    }
    pathlib.Path(cfg.meta_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def self_test() -> None:
    print_header("SELF TEST")
    base = "/tmp/kaggle_libero_selftest"
    if os.path.exists(base):
        shutil.rmtree(base)
    pathlib.Path(base).mkdir(parents=True, exist_ok=True)

    cfg = RunnerConfig(
        repo_dir=os.path.join(base, "LIBERO"),
        venv_dir=os.path.join(base, "venv"),
        libero_config_path=os.path.join(base, ".libero"),
        datasets_root=os.path.join(base, "datasets"),
        runs_root=os.path.join(base, "runs"),
        meta_path=os.path.join(base, "meta.json"),
        src_root=os.path.join(base, "LIBERO", "experiments"),
        snap_root=os.path.join(base, "snapshots"),
        zip_path=os.path.join(base, "snapshots.zip"),
        install_os_packages=False,
        rebuild_env=False,
        download_datasets=False,
        target_task_ids=[3, 4],
        package_name="selftest_bundle",
    )

    create_fake_meta(base, cfg)

    # Create fake experiment artifacts for snapshot testing.
    exp_root = pathlib.Path(cfg.src_root) / cfg.benchmark / "Sequential" / f"BCTransformerPolicy_seed{cfg.seed}" / "run_001"
    exp_root.mkdir(parents=True, exist_ok=True)
    for name in ["config.json", "task0_model.pth", "task0_auc.log", "result.pt", "ignore.bin"]:
        (exp_root / name).write_text("dummy", encoding="utf-8")

    snap = SnapshotManager(cfg.src_root, cfg.snap_root, cfg.zip_path, cfg.keep_exts, period_sec=1)
    copied = snap.snapshot_once()
    assert copied >= 4, f"expected at least 4 copied files, got {copied}"
    snap.zip_snapshots()
    assert os.path.exists(cfg.zip_path), "snapshot zip was not created"

    # Validate packaging logic with fake manifests.
    bundle_dir = pathlib.Path(base) / cfg.package_name
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "summary.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    zip_path = package_results(bundle_dir, str(pathlib.Path(base) / cfg.package_name))
    assert os.path.exists(zip_path), "bundle zip was not created"

    # Syntax-check launcher generation.
    launcher_path = write_launcher(
        target_task_id=3,
        py_path=sys.executable,
        repo_dir=cfg.repo_dir,
        benchmark_root=os.path.join(cfg.repo_dir, "libero", "libero"),
        datasets_root=cfg.datasets_root,
        run_dir=os.path.join(cfg.runs_root, "dummy_run"),
        cfg=cfg,
        stage_overrides=[],
    )
    compile(pathlib.Path(launcher_path).read_text(encoding="utf-8"), launcher_path, "exec")

    # Syntax-check this script.
    compile(pathlib.Path(__file__).read_text(encoding="utf-8"), __file__, "exec")
    print("Self-test passed.")


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kaggle LIBERO runner")
    parser.add_argument("--setup", action="store_true", help="Run environment setup")
    parser.add_argument("--run", action="store_true", help="Run tasks after setup")
    parser.add_argument("--self-test", action="store_true", help="Run a local self-test without Kaggle")
    parser.add_argument("--run-stage", default="full", choices=["canary0", "canary1", "canary2", "full"])
    parser.add_argument("--task-ids", default="3,4", help="Comma-separated zero-based task IDs")
    parser.add_argument("--eval-num-procs", type=int, default=1)
    parser.add_argument("--rebuild-env", action="store_true")
    parser.add_argument("--snapshot-period-sec", type=int, default=300)
    parser.add_argument("--package-name", default="libero_object_tasks_3_4_results")
    parser.add_argument("--skip-os-packages", action="store_true")
    parser.add_argument("--skip-dataset-download", action="store_true")
    parser.add_argument(
        "--repo-url",
        default="https://github.com/Lifelong-Robot-Learning/LIBERO.git",
        help="Git URL to clone when repo_dir does not exist",
    )
    parser.add_argument(
        "--extra-override",
        action="append",
        default=[],
        help="Extra Hydra override (repeat this flag for multiple overrides)",
    )
    return parser.parse_args()


def make_config_from_args(args: argparse.Namespace) -> RunnerConfig:
    task_ids = [int(x) for x in args.task_ids.split(",") if x.strip()]
    return RunnerConfig(
        repo_url=args.repo_url,
        run_stage=args.run_stage,
        target_task_ids=task_ids,
        eval_num_procs=args.eval_num_procs,
        rebuild_env=args.rebuild_env,
        snapshot_period_sec=args.snapshot_period_sec,
        package_name=args.package_name,
        extra_overrides=args.extra_override,
        install_os_packages=not args.skip_os_packages,
        download_datasets=not args.skip_dataset_download,
    )


def main() -> None:
    args = parse_args()

    if args.self_test:
        self_test()
        return

    if not args.setup and not args.run:
        raise SystemExit("Nothing to do. Use --setup, --run, or both.")

    cfg = make_config_from_args(args)
    snap = SnapshotManager(cfg.src_root, cfg.snap_root, cfg.zip_path, cfg.keep_exts, cfg.snapshot_period_sec)
    snap.start()

    try:
        if args.setup:
            setup_environment(cfg)
        if args.run:
            run_tasks(cfg)
    finally:
        # Final best-effort snapshot + zip.
        try:
            snap.snapshot_once()
            snap.zip_snapshots()
        finally:
            snap.stop()


if __name__ == "__main__":
    main()
