"""A deploy run: the recipe's stages in order, each one an `edgeinfer` command whose report
decides the next step (D-038, D-039).

    export -> convert -> validate -> push -> build -> parity -> accuracy -> perf -> publish

Producing stages (export, convert, build) are skipped when the command reuses an artifact
with the same content-addressed id. Gate checks are reused when an earlier run recorded a
passing check with the same key (D-040, `reuse.py`). The first failed stage stops the run.

The run's state is its deploy report, `artifacts/reports/deploy/<recipe>/<run_id>/
deploy.json`, rewritten after every stage, next to the reports of the commands it ran.
`--stage` runs one stage of an existing run, which is how the Airflow DAG drives it: the
state passes between tasks through that file and the content-addressed artifacts.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from edgeinfer.artifacts import ONNX_DIR, load_sidecar, sidecar_path
from edgeinfer.config import CONFIG_DIR, load_budgets, load_model_config
from edgeinfer.data.manifests import MANIFEST_DIR
from edgeinfer.golden import HASH_DIR
from edgeinfer.pipeline import registry
from edgeinfer.pipeline.config import Recipe, load_gates, recipe_hash
from edgeinfer.pipeline.executor import LocalExecutor, RemoteExecutor, rel
from edgeinfer.pipeline.reuse import check_key, find_reusable
from edgeinfer.reports import REPORTS_DIR, git_state, make_report, write_report

STAGES = ("export", "convert", "validate", "push", "build", "parity", "accuracy", "perf", "publish")
GATES = ("validate", "parity", "accuracy", "perf")

EXIT_PASSED, EXIT_BLOCKED, EXIT_ERROR = 0, 1, 2


@dataclass(frozen=True)
class Paths:
    """Where a run reads and writes. Defaults are the repo's; tests point them elsewhere."""

    reports_dir: Path = REPORTS_DIR
    onnx_root: Path = ONNX_DIR
    registry_dir: Path = registry.REGISTRY_DIR
    golden_dir: Path | None = None  # default: artifacts/golden/<model>
    calib_dir: Path | None = None  # default: artifacts/calib
    root: str | None = None  # Cityscapes root for local stages (default: $CITYSCAPES_ROOT)


@dataclass
class CheckSpec:
    """One gate check before it runs: the command, what it measures, and what it reads.
    `name` names the report in the run; `check` (default: `name`) goes in the key."""

    name: str
    executor: Any
    argv: list[str]
    artifact: dict[str, Any]
    settings: dict[str, Any]
    files: list[Path]
    check: str | None = None
    before: Callable[[], None] | None = None

    def key(self) -> dict[str, Any]:
        return check_key(self.check or self.name, self.artifact, self.settings, self.files)


class StageStop(Exception):
    """A stage ended the run: a failed check (exit 1) or an error (exit 2)."""

    def __init__(self, code: str, message: str, exit_code: int) -> None:
        super().__init__(message)
        self.code, self.exit_code = code, exit_code


def new_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def last_run(recipe: str, reports_dir: Path = REPORTS_DIR) -> dict[str, Any] | None:
    """The newest deploy report of `recipe`, or None."""
    runs = sorted((reports_dir / "deploy" / recipe).glob("*/deploy.json"))
    return json.loads(runs[-1].read_text()) if runs else None


def _config_file(name_or_path: str, subdir: Path) -> Path:
    path = Path(name_or_path)
    return path if path.suffix in (".yaml", ".yml") else subdir / f"{name_or_path}.yaml"


def _manifest_file(name_or_path: str) -> Path:
    path = Path(name_or_path)
    return path if path.suffix == ".txt" else MANIFEST_DIR / f"{name_or_path}.txt"


class Run:
    def __init__(
        self,
        recipe: Recipe,
        run_id: str,
        paths: Paths | None = None,
        gpu: Any = None,
        local: Any = None,
    ) -> None:
        self.recipe = recipe
        self.run_id = run_id
        self.paths = paths or Paths()
        self.cfg = load_model_config(recipe.model)
        self.gates = load_gates(recipe.gates)
        self.local = local or LocalExecutor(self.paths.root)
        self.gpu = gpu or (RemoteExecutor() if recipe.target == "trt" else self.local)
        self.dir = self.paths.reports_dir / "deploy" / recipe.name / run_id
        self.state_path = self.dir / "deploy.json"
        if self.state_path.is_file():
            report = json.loads(self.state_path.read_text())
            self.metrics, self.findings = report["metrics"], report["findings"]
        else:
            self.metrics = {"run_id": run_id, "stages": {}, "artifacts": {}, "published": None}
            self.findings: list[dict[str, str]] = []

    # --- state ---------------------------------------------------------------------------

    @property
    def stages(self) -> dict[str, dict[str, Any]]:
        return self.metrics["stages"]

    @property
    def artifacts(self) -> dict[str, dict[str, Any]]:
        return self.metrics["artifacts"]

    def _deployed(self) -> dict[str, Any] | None:
        return self.artifacts.get("engine" if self.recipe.target == "trt" else "onnx")

    def save(self) -> Path:
        deployed = self._deployed()
        done = self.metrics["published"] is not None
        report = make_report(
            tool="deploy",
            model=self.cfg.name,
            backend_meta={
                "backend": self.recipe.target,
                "precision": self.recipe.precision,
                "device": "pipeline",
                "artifact_id": deployed["id"] if deployed else "none",
            },
            inputs={
                "recipe": self.recipe.name,
                "recipe_hash": recipe_hash(self.recipe),
                "settings": self.recipe.to_dict(),
                "gates": self.gates.to_dict(),
            },
            metrics=self.metrics,
            findings=self.findings,
            passed=done and not any(f["severity"] == "error" for f in self.findings),
        )
        return write_report(report, self.state_path)

    def _accuracy_applies(self) -> bool:
        return self.recipe.precision in self.gates.accuracy.precisions

    # --- running commands ----------------------------------------------------------------

    def _report_path(self, name: str) -> Path:
        return self.dir / f"{name}.json"

    def _command(self, executor: Any, argv: list[str], name: str) -> tuple[int, dict | None]:
        """Run `argv` with its report at <run dir>/<name>.json; (exit code, report)."""
        out = self._report_path(name)
        code = executor.run([*argv, "--out", rel(out)])
        report = json.loads(out.read_text()) if out.is_file() else None
        if code not in (0, 1) or report is None:
            raise StageStop(
                "DEPLOY_STAGE_ERROR",
                f"`edgeinfer {argv[0]}` exited {code}"
                + ("" if report else " without writing a report"),
                EXIT_ERROR,
            )
        return code, report

    def _artifact(self, path: Path) -> dict[str, Any]:
        meta = load_sidecar(path)
        return {
            "path": rel(path),
            "id": meta["artifact_id"],
            "sha256": meta["sha256"],
            "precision": meta["precision"],
        }

    def _check(self, spec: CheckSpec) -> dict:
        """A gate check: reuse an earlier passing report with the same key, or run it."""
        key = spec.key()
        reuse = find_reusable(key, self.paths.reports_dir)
        record = {"name": spec.name, "key": key, "reason": reuse.reason}
        if reuse.report:
            report = json.loads((self.paths.reports_dir / reuse.report).read_text())
            return record | {
                "report": reuse.report,
                "status": "reused",
                "passed": True,
                "reused_from": reuse.run_id,
                "findings": report["findings"],
            }
        if spec.before:
            spec.before()
        code, report = self._command(spec.executor, spec.argv, spec.name)
        out = self._report_path(spec.name).relative_to(self.paths.reports_dir)
        return record | {
            "report": str(out),
            "status": "ran",
            "passed": code == 0 and report["passed"],
            "reused_from": None,
            "findings": report["findings"],
        }

    def _model_files(self) -> list[Path]:
        return [_config_file(self.recipe.model, CONFIG_DIR / "models")]

    # --- stages ----------------------------------------------------------------------------

    def stage_export(self) -> dict[str, Any]:
        # export's --out is the artifact root; it names its own report in --report-dir.
        report_dir = self.dir / "export"
        code = self.local.run(
            [
                "export",
                "--model",
                self.recipe.model,
                "--out",
                rel(self.paths.onnx_root),
                "--report-dir",
                rel(report_dir),
            ]
        )
        found = sorted(report_dir.glob("export__*.json"))
        if code != 0 or len(found) != 1:
            raise StageStop("DEPLOY_STAGE_ERROR", f"`edgeinfer export` exited {code}", EXIT_ERROR)
        report = json.loads(found[0].read_text())
        path = self.paths.onnx_root / self.cfg.name / report["metrics"]["file"]
        self.artifacts["fp32"] = self._artifact(path)
        if self.recipe.precision == "fp32":
            self.artifacts["onnx"] = self.artifacts["fp32"]
        return {
            "status": "skipped" if report["metrics"]["reused"] else "ran",
            "passed": True,
            "reports": [rel(found[0])],
        }

    def stage_convert(self) -> dict[str, Any]:
        r, fp32 = self.recipe, self.artifacts["fp32"]["path"]
        if r.precision == "fp32":
            return {"status": "skipped", "passed": True, "reason": "precision fp32"}
        common = [
            "--model",
            r.model,
            "--artifact",
            fp32,
            "--out-root",
            rel(self.paths.onnx_root),
            *self.local.dataset_args(),
        ]
        if self.paths.calib_dir:
            common += ["--calib-dir", rel(self.paths.calib_dir)]
        if r.precision == "fp16":
            argv = ["fp16", *common, "--fp16-config", r.fp16]
        else:
            argv = ["quantize", *common, "--quant-config", r.quant.config]
            if r.quant.exclusions:
                argv += ["--exclusions", r.quant.exclusions]
            if r.quant.groups:
                argv += ["--groups", r.quant.groups]
            if r.precision == "int8_fp16":
                argv += ["--float-dtype", "fp16"]
        code, report = self._command(self.local, argv, "convert")
        if code != 0:
            raise StageStop("DEPLOY_STAGE_FAILED", f"`edgeinfer {argv[0]}` failed", EXIT_BLOCKED)
        self.artifacts["onnx"] = self._artifact(
            self.paths.onnx_root / self.cfg.name / report["metrics"]["file"]
        )
        if self.artifacts["onnx"]["precision"] != r.precision:
            raise StageStop(
                "DEPLOY_STAGE_ERROR",
                f"convert made a {self.artifacts['onnx']['precision']} model for a "
                f"{r.precision} recipe",
                EXIT_ERROR,
            )
        return {
            "status": "skipped" if report["metrics"]["reused"] else "ran",
            "passed": True,
            "reports": ["convert.json"],
        }

    def _validate_specs(self) -> list[CheckSpec]:
        onnx = self.artifacts["onnx"]
        argv = ["validate", "--model", self.recipe.model, "--artifact", onnx["path"]]
        files = [*self._model_files(), CONFIG_DIR / "supported_ops.yaml"]
        return [CheckSpec("validate", self.local, argv, onnx, {}, files)]

    def stage_validate(self) -> dict[str, Any]:
        return {"checks": [self._check(spec) for spec in self._validate_specs()]}

    def _pushed(self) -> list[str]:
        names = ["onnx", "fp32"] if self._accuracy_applies() else ["onnx"]
        paths = {self.artifacts[n]["path"] for n in names}
        return sorted(p for path in paths for p in (path, rel(sidecar_path(Path(path)))))

    def stage_push(self) -> dict[str, Any]:
        if self.recipe.target != "trt":
            return {"status": "skipped", "passed": True, "reason": "local target"}
        code = self.gpu.push(self._pushed())
        if code != 0:
            raise StageStop("DEPLOY_STAGE_ERROR", f"push-artifacts exited {code}", EXIT_ERROR)
        return {"status": "ran", "passed": True}  # rsync copies only what changed

    def _build(self, name: str, onnx: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        argv = [
            "build",
            "--model",
            self.recipe.model,
            "--artifact",
            onnx["path"],
            "--build-config",
            self.recipe.build,
        ]
        code, report = self._command(self.gpu, argv, name)
        if code != 0:
            raise StageStop(
                "DEPLOY_STAGE_FAILED", f"`edgeinfer build` failed ({name})", EXIT_BLOCKED
            )
        m = report["metrics"]
        engine = {
            "path": f"artifacts/engines/{self.cfg.name}/{m['file']}",
            "id": report["artifact_id"],
            "sha256": m["sha256"],
            "precision": report["precision"],
        }
        return engine, m["reused"]

    def stage_build(self) -> dict[str, Any]:
        if self.recipe.target != "trt":
            return {"status": "skipped", "passed": True, "reason": "local target"}
        self.artifacts["engine"], reused = self._build("build", self.artifacts["onnx"])
        reports = ["build.json"]
        if self._accuracy_applies():
            self.artifacts["fp32_engine"], ref_reused = self._build(
                "build_fp32", self.artifacts["fp32"]
            )
            reused, reports = reused and ref_reused, [*reports, "build_fp32.json"]
        return {"status": "skipped" if reused else "ran", "passed": True, "reports": reports}

    def _parity_specs(self) -> list[CheckSpec]:
        g, art = self.gates.parity, self._deployed()
        argv = [
            "parity",
            "--model",
            self.recipe.model,
            "--backend",
            self.recipe.target,
            "--artifact",
            art["path"],
            "--inputs",
            g.inputs,
            "--manifest",
            g.manifest,
            *self.gpu.dataset_args(),
        ]
        if self.paths.golden_dir:
            argv += ["--golden-dir", rel(self.paths.golden_dir)]
        files = [*self._model_files(), CONFIG_DIR / "tolerances.yaml", _manifest_file(g.manifest)]
        hashes = HASH_DIR / f"{self.cfg.name}.sha256"
        files += [hashes] if hashes.is_file() else []
        return [CheckSpec("parity", self.gpu, argv, art, {"parity": asdict(g)}, files)]

    def stage_parity(self) -> dict[str, Any]:
        return {"checks": [self._check(spec) for spec in self._parity_specs()]}

    def _eval_spec(self, name: str, art: dict[str, Any]) -> CheckSpec:
        g = self.gates.accuracy
        argv = [
            "eval",
            "--model",
            self.recipe.model,
            "--backend",
            self.recipe.target,
            "--artifact",
            art["path"],
            "--manifest",
            g.manifest,
            *self.gpu.dataset_args(),
        ]
        # Keyed as "eval" whatever the report is called: the FP32 reference eval of one
        # artifact is the same measurement in every recipe that uses it.
        files = [*self._model_files(), _manifest_file(g.manifest)]
        return CheckSpec(name, self.gpu, argv, art, {"manifest": g.manifest}, files, check="eval")

    def _accuracy_specs(self) -> list[CheckSpec]:
        if not self._accuracy_applies():
            return []
        ref = self.artifacts["fp32_engine" if self.recipe.target == "trt" else "fp32"]
        return [self._eval_spec("eval_fp32", ref), self._eval_spec("eval", self._deployed())]

    def stage_accuracy(self) -> dict[str, Any]:
        if not self._accuracy_applies():
            return {
                "status": "skipped",
                "passed": True,
                "reason": f"precision {self.recipe.precision} is gated by golden parity",
            }
        checks = [self._check(spec) for spec in self._accuracy_specs()]
        if not all(c["passed"] for c in checks):
            return {"checks": checks}
        ref_r, cand_r = (
            json.loads((self.paths.reports_dir / c["report"]).read_text()) for c in checks
        )
        if ref_r["metrics"]["n_images"] != cand_r["metrics"]["n_images"]:
            raise StageStop(
                "DEPLOY_STAGE_ERROR", "the two evals scored different images", EXIT_ERROR
            )
        budget = load_budgets()["accuracy"]["miou_drop_max_points"]
        drop = 100 * (ref_r["metrics"]["miou"] - cand_r["metrics"]["miou"])
        verdict = {
            "miou_fp32": ref_r["metrics"]["miou"],
            "miou": cand_r["metrics"]["miou"],
            "drop_points": drop,
            "budget_points": budget,
            "n_images": cand_r["metrics"]["n_images"],
        }
        return {
            "checks": checks,
            "passed": drop <= budget,
            "verdict": verdict,
            "message": f"mIoU drop {drop:.3f} points (budget {budget})",
        }

    def _perf_specs(self) -> list[CheckSpec]:
        from edgeinfer.perf.probe import load_perf_config

        g, art = self.gates.perf, self._deployed()
        perf_file = _config_file(g.perf_config, CONFIG_DIR)
        pcfg = load_perf_config(perf_file)
        argv = [
            "perf",
            "--model",
            self.recipe.model,
            "--backend",
            self.recipe.target,
            "--artifact",
            art["path"],
            "--perf-config",
            rel(perf_file),
            *self.gpu.dataset_args(),
        ]
        files = [
            *self._model_files(),
            perf_file,
            CONFIG_DIR / "budgets.yaml",
            _manifest_file(pcfg.manifest),
        ]

        def lock_clocks() -> None:  # D-027; the probe fails if the clock isn't locked
            if self.recipe.target == "trt" and pcfg.gpu_clock_mhz:
                mhz = pcfg.gpu_clock_mhz
                if self.gpu.host(f"sudo nvidia-smi --lock-gpu-clocks={mhz},{mhz}") != 0:
                    raise StageStop("DEPLOY_STAGE_ERROR", "couldn't lock the GPU clock", EXIT_ERROR)

        return [
            CheckSpec("perf", self.gpu, argv, art, {"perf": asdict(g)}, files, before=lock_clocks)
        ]

    def stage_perf(self) -> dict[str, Any]:
        (check,) = [self._check(spec) for spec in self._perf_specs()]
        enforced = not any(f["code"] == "PERF_DEV_HOST" for f in check["findings"])
        return {"checks": [check], "enforced": enforced}

    def stage_publish(self) -> dict[str, Any]:
        sha, dirty = git_state()
        if dirty:
            raise StageStop(
                "DEPLOY_DIRTY_TREE",
                "uncommitted changes: a deployment must trace to a commit",
                EXIT_BLOCKED,
            )
        enforced = all(self.stages[g].get("enforced", True) for g in GATES)
        alias = "production" if enforced else "dev"
        art = self._deployed()
        location = (
            registry.INSTANCE_PREFIX + art["path"] if self.recipe.target == "trt" else art["path"]
        )
        record = {
            "model": self.cfg.name,
            "artifact_id": art["id"],
            "sha256": art["sha256"],
            "precision": self.recipe.precision,
            "target": self.recipe.target,
            "location": location,
            "recipe": self.recipe.name,
            "recipe_hash": recipe_hash(self.recipe),
            "run_id": self.run_id,
            "deploy_report": str(self.state_path.relative_to(self.paths.reports_dir)),
            "git_sha": sha,
            "gate_reports": {
                g: [c["report"] for c in self.stages[g].get("checks", [])] for g in GATES
            },
        }
        version, new = registry.publish(record, alias, self.paths.registry_dir)
        self.metrics["published"] = {"version": version, "alias": alias, "new_version": new}
        self.findings.append(
            {
                "severity": "info",
                "code": "REGISTRY_PUBLISHED",
                "message": f"{self.cfg.name} v{version} -> {alias}"
                + ("" if new else " (already registered)"),
            }
        )
        return {"status": "ran", "passed": True, "version": version, "alias": alias}

    # --- dry run ---------------------------------------------------------------------------

    def _needs(self, gate: str) -> list[str]:
        """Artifacts a gate's checks measure."""
        deployed = "engine" if self.recipe.target == "trt" else "onnx"
        ref = "fp32_engine" if self.recipe.target == "trt" else "fp32"
        return {
            "validate": ["onnx"],
            "parity": [deployed],
            "perf": [deployed],
            "accuracy": [ref, deployed],
        }[gate]

    def plan(self, previous: dict[str, Any] | None) -> list[tuple[str, str, str]]:
        """What a run would do, without running anything: (stage, action, why).

        Gate checks are judged with the artifacts of `previous` (the recipe's last run), so
        the answer assumes the producing stages make the same artifacts again; they decide
        that themselves when they run."""
        if previous:
            self.metrics["artifacts"] = dict(previous["metrics"]["artifacts"])
        trt = self.recipe.target == "trt"
        rows: list[tuple[str, str, str]] = []
        for name in STAGES:
            if name in ("push", "build") and not trt:
                rows.append((name, "skip", "local target"))
            elif name == "convert" and self.recipe.precision == "fp32":
                rows.append((name, "skip", "precision fp32"))
            elif name == "accuracy" and not self._accuracy_applies():
                rows.append((name, "skip", f"precision {self.recipe.precision}"))
            elif name in ("export", "convert", "build"):
                rows.append((name, "run", "skipped by the command if its output id exists"))
            elif name == "push":
                rows.append((name, "run", "rsync copies only files that changed"))
            elif name == "publish":
                rows.append(
                    (name, "run", f"to {'production' if trt else 'dev'} if every gate passes")
                )
            else:
                missing = [a for a in self._needs(name) if a not in self.artifacts]
                if missing:
                    rows.append((name, "run", f"{missing[0]} not known before this run makes it"))
                    continue
                for spec in getattr(self, f"_{name}_specs")():
                    reuse = find_reusable(spec.key(), self.paths.reports_dir)
                    action = "reuse" if reuse.report else "run"
                    rows.append((f"{name}/{spec.name}", action, reuse.reason))
        return rows

    # --- driving ---------------------------------------------------------------------------

    def _finish_entry(self, name: str, entry: dict[str, Any], seconds: float) -> dict:
        checks = entry.get("checks")
        if checks is not None:
            entry.setdefault("passed", all(c["passed"] for c in checks))
            entry.setdefault(
                "status", "reused" if all(c["status"] == "reused" for c in checks) else "ran"
            )
        if entry["passed"] is False:
            entry["status"] = "failed"
        entry["seconds"] = round(seconds, 1)
        entry["where"] = (
            "instance"
            if (name in ("push", "build") and self.recipe.target == "trt")
            or (name in ("parity", "accuracy", "perf") and self.gpu.remote)
            else "local"
        )
        return entry

    def run_stage(self, name: str) -> int:
        """Run one stage; earlier stages must have passed. Returns the stage's exit code."""
        earlier = STAGES[: STAGES.index(name)]
        missing = [s for s in earlier if self.stages.get(s, {}).get("passed") is not True]
        if missing:
            print(f"error: stage {name} needs {missing[0]} to have passed in run {self.run_id}")
            return EXIT_ERROR
        if self.stages.get(name, {}).get("passed") is True:
            print(f"{name}: already passed in run {self.run_id}")
            return EXIT_PASSED
        t0 = time.perf_counter()
        try:
            entry = getattr(self, f"stage_{name}")()
            exit_code = EXIT_PASSED
        except StageStop as e:
            entry, exit_code = (
                {
                    "status": "error" if e.exit_code == EXIT_ERROR else "failed",
                    "passed": False,
                    "message": str(e),
                },
                e.exit_code,
            )
            self.findings.append(
                {"severity": "error", "code": e.code, "message": str(e), "where": name}
            )
        entry = self._finish_entry(name, entry, time.perf_counter() - t0)
        self.stages[name] = entry
        if entry["status"] == "failed" and exit_code == EXIT_PASSED:
            exit_code = EXIT_BLOCKED
            code = "DEPLOY_GATE_FAILED" if name in GATES else "DEPLOY_STAGE_FAILED"
            self.findings.append(
                {
                    "severity": "error",
                    "code": code,
                    "where": name,
                    "message": entry.get("message") or self._failed_codes(entry),
                }
            )
        if entry.get("enforced") is False:
            self.findings.append(
                {
                    "severity": "warn",
                    "code": "DEPLOY_GATE_NOT_ENFORCED",
                    "where": name,
                    "message": "measured on a development host: "
                    "not gated, so this deployment can only reach `dev`",
                }
            )
        self.save()
        print(
            f"{name}: {entry['status']} ({entry['seconds']} s)"
            + (f" - {entry['message']}" if entry.get("message") else "")
        )
        return exit_code

    @staticmethod
    def _failed_codes(entry: dict[str, Any]) -> str:
        codes = [
            f"{c['name']}: {f['code']}"
            for c in entry.get("checks", [])
            for f in c["findings"]
            if f["severity"] == "error"
        ]
        return "; ".join(codes) or "check failed"

    def run_all(self) -> int:
        for name in STAGES:
            if self.stages.get(name, {}).get("passed") is True:
                continue
            code = self.run_stage(name)
            if code != EXIT_PASSED:
                for later in STAGES[STAGES.index(name) + 1 :]:
                    self.stages[later] = {"status": "not_run", "passed": None}
                self.save()
                return code
        return EXIT_PASSED
