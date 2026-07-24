#!/usr/bin/env python3
"""Retry formal ABBA/BAAB runtime using a proven evaluator artifact incarnation.

This entry point deliberately cannot prepare, clone, emit, or build artifacts.
It acquires an already-existing evaluator slot, proves that its candidate
artifacts are the exact incarnation which produced a retryable runtime
manifest after all attribution and function gates, and then delegates the
runtime, promotion, scoring, and manifest work back to :mod:`evaluator`.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import sys
import time
from types import ModuleType
from typing import Any, Iterator, Mapping


TASK_ROOT = Path(__file__).resolve().parent


def _load_evaluator() -> ModuleType:
    module_name = "_simpletes_grhsim_simtop_retry_evaluator"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    path = TASK_ROOT / "evaluator.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load evaluator module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


evaluator = _load_evaluator()


_HEX_32_RE = re.compile(r"^[0-9a-f]{32}$")
_HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")
_CANDIDATE_PROOF_FIELDS = {
    "schema_version",
    "proof_version",
    "proof_id",
    "created_time_ns",
    "candidate_digest",
    "patch_sha256",
    "canonical_enable_options",
    "parent_commit",
    "wolvrix_commit",
    "repo",
    "generated_fingerprint",
    "build_config_fingerprint",
    "toolchain_fingerprint",
    "artifacts",
    "env_sh",
}
_ATTEMPT_COMPLETE_FIELDS = {
    "schema_version",
    "attempt_id",
    "created_time_ns",
    "candidate_digest",
    "candidate_proof_id",
    "candidate_proof_sha256",
    "runtime_result_sha256",
    "evaluation_sha256",
}


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise evaluator.InfrastructureError(f"{label} is not a regular file: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=evaluator._reject_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, evaluator.CandidateError) as error:
        raise evaluator.InfrastructureError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise evaluator.InfrastructureError(f"{label} is not a JSON object: {path}")
    return value


def _require_regular_nonempty(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise evaluator.InfrastructureError(f"{label} is not a regular file: {path}")
    if path.stat().st_size <= 0:
        raise evaluator.InfrastructureError(f"{label} is empty: {path}")


def _artifact_sha256(artifacts: Any) -> dict[str, str]:
    return {
        "binary": evaluator._sha256_file(artifacts.binary),
        "image": evaluator._sha256_file(artifacts.image),
        "nemu": evaluator._sha256_file(artifacts.nemu),
    }


def _require_exact_fields(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise evaluator.InfrastructureError(
            f"{label} fields differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _is_int(value: Any) -> bool:
    return type(value) is int


def _is_float(value: Any) -> bool:
    return type(value) is float


def _load_verified_control_artifacts(slot: Any) -> Any:
    """Load the cached control or fail; unlike evaluator, never rebuild it."""

    marker_path = slot.results_dir / "control_artifacts.json"
    raw = _read_json_object(marker_path, "control artifact marker")
    env_sh = slot.control_repo / "env.sh"
    _require_regular_nonempty(env_sh, "control env.sh")
    evaluator._verify_pinned_repo(slot.control_repo, env_sh)

    try:
        artifacts = evaluator.BuildArtifacts(
            name=str(raw["name"]),
            repo=Path(raw["repo"]),
            binary=Path(raw["binary"]),
            image=Path(raw["image"]),
            nemu=Path(raw["nemu"]),
            generated_fingerprint=str(raw["generated_fingerprint"]),
            build_log=Path(raw["build_log"]) if raw.get("build_log") else None,
            build_config_fingerprint=str(raw["build_config_fingerprint"]),
            toolchain_fingerprint=str(raw["toolchain_fingerprint"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise evaluator.InfrastructureError(
            f"malformed control artifact marker: {error}"
        ) from error

    binary, image, nemu = evaluator._artifact_paths(slot.control_repo)
    jobs = max(1, int(os.environ.get("GRHSIM_BUILD_JOBS", "4")))
    expected_build = evaluator._build_config_fingerprint({}, jobs=jobs)
    expected_toolchain = evaluator._toolchain_fingerprint(slot.control_repo, env_sh)
    expected_sha256 = _artifact_sha256(
        evaluator.BuildArtifacts(
            name="control",
            repo=slot.control_repo,
            binary=binary,
            image=image,
            nemu=nemu,
            generated_fingerprint=artifacts.generated_fingerprint,
        )
    )
    checks = {
        "schema_version": _is_int(raw.get("schema_version"))
        and raw.get("schema_version") == evaluator.CONTROL_MARKER_SCHEMA_VERSION,
        "name": type(raw.get("name")) is str and artifacts.name == "control",
        "repo": type(raw.get("repo")) is str and artifacts.repo == slot.control_repo,
        "parent_commit": type(raw.get("parent_commit")) is str
        and raw.get("parent_commit") == evaluator.PINNED_PARENT_COMMIT,
        "wolvrix_commit": type(raw.get("wolvrix_commit")) is str
        and raw.get("wolvrix_commit") == evaluator.PINNED_WOLVRIX_COMMIT,
        "binary_path": artifacts.binary == binary,
        "image_path": artifacts.image == image,
        "nemu_path": artifacts.nemu == nemu,
        "generated_fingerprint": evaluator.generated_fingerprint(slot.control_repo)
        == artifacts.generated_fingerprint,
        "build_config_fingerprint": artifacts.build_config_fingerprint == expected_build,
        "toolchain_fingerprint": artifacts.toolchain_fingerprint == expected_toolchain,
        "artifact_sha256": raw.get("artifact_sha256") == expected_sha256,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise evaluator.InfrastructureError(
            "cached control failed runtime-only proof: " + ", ".join(failed)
        )
    return artifacts


def _artifact_entry(raw: Mapping[str, Any], name: str) -> tuple[Path, str]:
    value = raw.get(name)
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise evaluator.InfrastructureError(f"malformed candidate proof artifact: {name}")
    path = value.get("path")
    sha256 = value.get("sha256")
    if type(path) is not str or type(sha256) is not str or not _HEX_64_RE.fullmatch(sha256):
        raise evaluator.InfrastructureError(f"malformed candidate proof artifact: {name}")
    return Path(path), sha256


def _load_candidate_proof(candidate: Any, slot: Any, control: Any) -> tuple[Any, str]:
    proof_path = evaluator._candidate_proof_path(slot, candidate)
    raw = _read_json_object(proof_path, "candidate artifact proof")
    _require_exact_fields(raw, _CANDIDATE_PROOF_FIELDS, "candidate artifact proof")
    proof_sha256 = evaluator._sha256_file(proof_path)
    proof_id = raw.get("proof_id")
    created_time_ns = raw.get("created_time_ns")
    if type(proof_id) is not str or not _HEX_32_RE.fullmatch(proof_id):
        raise evaluator.InfrastructureError("candidate artifact proof has invalid proof_id")
    if not _is_int(created_time_ns) or created_time_ns <= 0:
        raise evaluator.InfrastructureError("candidate artifact proof has invalid created_time_ns")
    typed_fields = {
        "schema_version": _is_int(raw.get("schema_version")),
        "proof_version": _is_int(raw.get("proof_version")),
        "candidate_digest": type(raw.get("candidate_digest")) is str
        and bool(_HEX_64_RE.fullmatch(raw.get("candidate_digest", ""))),
        "patch_sha256": type(raw.get("patch_sha256")) is str
        and bool(_HEX_64_RE.fullmatch(raw.get("patch_sha256", ""))),
        "canonical_enable_options": type(raw.get("canonical_enable_options")) is str,
        "parent_commit": type(raw.get("parent_commit")) is str,
        "wolvrix_commit": type(raw.get("wolvrix_commit")) is str,
        "repo": type(raw.get("repo")) is str,
        "generated_fingerprint": type(raw.get("generated_fingerprint")) is str
        and bool(_HEX_64_RE.fullmatch(raw.get("generated_fingerprint", ""))),
        "build_config_fingerprint": type(raw.get("build_config_fingerprint")) is str
        and bool(_HEX_64_RE.fullmatch(raw.get("build_config_fingerprint", ""))),
        "toolchain_fingerprint": type(raw.get("toolchain_fingerprint")) is str
        and bool(_HEX_64_RE.fullmatch(raw.get("toolchain_fingerprint", ""))),
    }
    malformed = sorted(name for name, passed in typed_fields.items() if not passed)
    if malformed:
        raise evaluator.InfrastructureError(
            "candidate artifact proof has malformed fields: " + ", ".join(malformed)
        )

    candidate_env = slot.candidate_repo / "env.sh"
    control_env = slot.control_repo / "env.sh"
    _require_regular_nonempty(candidate_env, "candidate env.sh")
    _require_regular_nonempty(control_env, "control env.sh")
    candidate_env_sha256 = evaluator._sha256_file(candidate_env)
    control_env_sha256 = evaluator._sha256_file(control_env)
    env_raw = raw.get("env_sh")
    expected_env = {
        "candidate_path": str(candidate_env),
        "candidate_sha256": candidate_env_sha256,
        "control_path": str(control_env),
        "control_sha256": control_env_sha256,
    }
    if not isinstance(env_raw, dict) or env_raw != expected_env:
        raise evaluator.InfrastructureError("candidate artifact proof env.sh binding differs")
    if candidate_env_sha256 != control_env_sha256:
        raise evaluator.InfrastructureError("candidate and control env.sh differ")

    evaluator._verify_pinned_repo(slot.candidate_repo, candidate_env)
    binary, image, nemu = evaluator._artifact_paths(slot.candidate_repo)
    artifacts_raw = raw.get("artifacts")
    if not isinstance(artifacts_raw, dict) or set(artifacts_raw) != {
        "binary",
        "image",
        "nemu",
    }:
        raise evaluator.InfrastructureError("malformed candidate artifact proof artifacts")
    actual_paths = {"binary": binary, "image": image, "nemu": nemu}
    for name, actual_path in actual_paths.items():
        marker_path, marker_sha256 = _artifact_entry(artifacts_raw, name)
        if marker_path != actual_path:
            raise evaluator.InfrastructureError(
                f"candidate artifact proof {name} path differs"
            )
        if evaluator._sha256_file(actual_path) != marker_sha256:
            raise evaluator.InfrastructureError(
                f"candidate artifact proof {name} SHA-256 differs"
            )

    generated = evaluator.generated_fingerprint(slot.candidate_repo)
    jobs = max(1, int(os.environ.get("GRHSIM_BUILD_JOBS", "4")))
    build_config = evaluator._build_config_fingerprint(candidate.enable_options, jobs=jobs)
    toolchain = evaluator._toolchain_fingerprint(slot.candidate_repo, candidate_env)
    patch_path = slot.root / "candidate.patch"
    _require_regular_nonempty(patch_path, "candidate patch")
    try:
        stored_patch = patch_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise evaluator.InfrastructureError(f"cannot read candidate patch: {error}") from error
    expected_scalar = {
        "schema_version": evaluator.CANDIDATE_PROOF_SCHEMA_VERSION,
        "proof_version": evaluator.CANDIDATE_PROOF_VERSION,
        "candidate_digest": candidate.digest,
        "patch_sha256": hashlib.sha256(candidate.patch.encode("utf-8")).hexdigest(),
        "canonical_enable_options": evaluator._canonical_enable_options(
            candidate.enable_options
        ),
        "parent_commit": evaluator.PINNED_PARENT_COMMIT,
        "wolvrix_commit": evaluator.PINNED_WOLVRIX_COMMIT,
        "repo": str(slot.candidate_repo.resolve()),
        "generated_fingerprint": generated,
        "build_config_fingerprint": build_config,
        "toolchain_fingerprint": toolchain,
    }
    failed = sorted(name for name, expected in expected_scalar.items() if raw.get(name) != expected)
    if failed:
        raise evaluator.InfrastructureError(
            "candidate artifact proof binding differs: " + ", ".join(failed)
        )
    if stored_patch != candidate.patch:
        raise evaluator.CandidateError(
            "candidate.patch does not exactly match the requested structured candidate"
        )

    if evaluator._sha256_file(image) != evaluator._sha256_file(control.image):
        raise evaluator.InfrastructureError("candidate and control images differ")
    if evaluator._sha256_file(nemu) != evaluator._sha256_file(control.nemu):
        raise evaluator.InfrastructureError("candidate and control NEMU artifacts differ")
    if toolchain != control.toolchain_fingerprint:
        raise evaluator.InfrastructureError("candidate and control toolchains differ")
    enabled = evaluator.BuildArtifacts(
        name=f"candidate_{candidate.digest[:12]}_enabled",
        repo=slot.candidate_repo,
        binary=binary,
        image=image,
        nemu=nemu,
        generated_fingerprint=generated,
        build_log=slot.results_dir / f"candidate_{candidate.digest[:12]}_enabled_build.log",
        build_config_fingerprint=build_config,
        toolchain_fingerprint=toolchain,
        candidate_proof_id=proof_id,
        candidate_proof_sha256=proof_sha256,
    )
    return enabled, proof_sha256


def _validate_attempt_document_identity(
    value: Mapping[str, Any],
    candidate: Any,
    attempt_id: str,
    proof_id: str,
    proof_sha256: str,
    label: str,
) -> None:
    checks = {
        "attempt_id": type(value.get("attempt_id")) is str
        and value.get("attempt_id") == attempt_id,
        "candidate_digest": type(value.get("candidate_digest")) is str
        and value.get("candidate_digest") == candidate.digest[:16],
        "candidate_digest_full": type(value.get("candidate_digest_full")) is str
        and value.get("candidate_digest_full") == candidate.digest,
        "candidate_proof_id": type(value.get("candidate_proof_id")) is str
        and value.get("candidate_proof_id") == proof_id,
        "candidate_proof_sha256": type(value.get("candidate_proof_sha256")) is str
        and value.get("candidate_proof_sha256") == proof_sha256,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise evaluator.InfrastructureError(
            f"{label} identity differs: " + ", ".join(failed)
        )


def _load_latest_complete_attempt(
    candidate: Any, slot: Any, proof_id: str, proof_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    attempts_root = slot.results_dir / "attempts" / candidate.digest
    if attempts_root.is_symlink() or not attempts_root.is_dir():
        raise evaluator.InfrastructureError(
            f"candidate has no immutable evaluation-attempt directory: {attempts_root}"
        )
    matching: list[tuple[int, str, dict[str, Any], dict[str, Any]]] = []
    for attempt_dir in sorted(attempts_root.iterdir()):
        if attempt_dir.is_symlink():
            raise evaluator.InfrastructureError(
                f"evaluation attempt may not be a symlink: {attempt_dir}"
            )
        if not attempt_dir.is_dir():
            raise evaluator.InfrastructureError(
                f"evaluation attempt entry is not a directory: {attempt_dir}"
            )
        complete_path = attempt_dir / "complete.json"
        if not complete_path.exists():
            continue
        complete = _read_json_object(complete_path, "evaluation attempt completion")
        _require_exact_fields(
            complete, _ATTEMPT_COMPLETE_FIELDS, "evaluation attempt completion"
        )
        attempt_id = complete.get("attempt_id")
        created_time_ns = complete.get("created_time_ns")
        checks = {
            "schema_version": _is_int(complete.get("schema_version"))
            and complete.get("schema_version")
            == evaluator.EVALUATION_ATTEMPT_SCHEMA_VERSION,
            "attempt_id": type(attempt_id) is str and attempt_id == attempt_dir.name,
            "created_time_ns": _is_int(created_time_ns) and created_time_ns > 0,
            "candidate_digest": type(complete.get("candidate_digest")) is str
            and complete.get("candidate_digest") == candidate.digest,
            "candidate_proof_id": type(complete.get("candidate_proof_id")) is str,
            "candidate_proof_sha256": type(complete.get("candidate_proof_sha256"))
            is str
            and bool(_HEX_64_RE.fullmatch(complete.get("candidate_proof_sha256", ""))),
            "runtime_result_sha256": type(complete.get("runtime_result_sha256")) is str
            and bool(_HEX_64_RE.fullmatch(complete.get("runtime_result_sha256", ""))),
            "evaluation_sha256": type(complete.get("evaluation_sha256")) is str
            and bool(_HEX_64_RE.fullmatch(complete.get("evaluation_sha256", ""))),
        }
        failed = sorted(name for name, passed in checks.items() if not passed)
        if failed:
            raise evaluator.InfrastructureError(
                "malformed complete evaluation attempt: " + ", ".join(failed)
            )
        runtime_path = attempt_dir / "runtime_result.json"
        evaluation_path = attempt_dir / "evaluation.json"
        if evaluator._sha256_file(runtime_path) != complete["runtime_result_sha256"]:
            raise evaluator.InfrastructureError("immutable runtime-result SHA-256 differs")
        if evaluator._sha256_file(evaluation_path) != complete["evaluation_sha256"]:
            raise evaluator.InfrastructureError("immutable evaluation SHA-256 differs")
        runtime = _read_json_object(runtime_path, "immutable runtime result")
        evaluation = _read_json_object(evaluation_path, "immutable evaluation")
        if not _is_int(runtime.get("schema_version")) or runtime.get(
            "schema_version"
        ) != evaluator.RUNTIME_RESULT_SCHEMA_VERSION:
            raise evaluator.InfrastructureError("runtime-result schema_version is invalid")
        if not _is_int(evaluation.get("schema_version")) or evaluation.get(
            "schema_version"
        ) != evaluator.EVALUATION_RESULT_SCHEMA_VERSION:
            raise evaluator.InfrastructureError("evaluation schema_version is invalid")
        _validate_attempt_document_identity(
            runtime,
            candidate,
            attempt_id,
            complete["candidate_proof_id"],
            complete["candidate_proof_sha256"],
            "runtime result",
        )
        _validate_attempt_document_identity(
            evaluation,
            candidate,
            attempt_id,
            complete["candidate_proof_id"],
            complete["candidate_proof_sha256"],
            "evaluation",
        )
        if (
            complete["candidate_proof_id"] == proof_id
            and complete["candidate_proof_sha256"] == proof_sha256
        ):
            matching.append((created_time_ns, attempt_id, runtime, evaluation))

    if not matching:
        raise evaluator.InfrastructureError(
            "no complete immutable evaluation attempt matches the current candidate proof"
        )
    _created, _attempt_id, runtime, prior = max(matching, key=lambda row: (row[0], row[1]))
    expected_files = len(evaluator.validate_patch(candidate.patch, allow_empty=True))
    evaluation_checks = {
        "valid_candidate": type(prior.get("valid_candidate")) is int
        and prior.get("valid_candidate") == 0,
        "validity": _is_float(prior.get("validity")) and prior.get("validity") == 0.0,
        "infrastructure_retry": _is_float(prior.get("infrastructure_retry"))
        and prior.get("infrastructure_retry") == 1.0,
        "retryable_infra": _is_float(prior.get("retryable_infra"))
        and prior.get("retryable_infra") == 1.0,
        "candidate_files": type(prior.get("candidate_files")) is int
        and prior.get("candidate_files") == expected_files,
        "enable_option_count": type(prior.get("enable_option_count")) is int
        and prior.get("enable_option_count") == len(candidate.enable_options),
        "parent_commit": type(prior.get("parent_commit")) is str
        and prior.get("parent_commit") == evaluator.PINNED_PARENT_COMMIT[:12],
        "wolvrix_commit": type(prior.get("wolvrix_commit")) is str
        and prior.get("wolvrix_commit") == evaluator.PINNED_WOLVRIX_COMMIT[:12],
    }
    runtime_checks = {
        "valid": runtime.get("valid") is False,
        "infrastructure_retry": runtime.get("infrastructure_retry") is True,
        "retryable_infra": runtime.get("retryable_infra") is True,
    }
    failed = sorted(
        [f"evaluation.{name}" for name, passed in evaluation_checks.items() if not passed]
        + [f"runtime.{name}" for name, passed in runtime_checks.items() if not passed]
    )
    if failed:
        raise evaluator.CandidateError(
            "latest complete attempt is not a strictly typed retryable runtime: "
            + ", ".join(failed)
        )
    return runtime, prior


def prepare_reused_artifacts(candidate: Any, slot: Any) -> tuple[Any, Any]:
    """Return previously gated artifacts, or fail before starting the runtime."""

    if candidate.is_control:
        raise evaluator.CandidateError("runtime-only retry requires a non-control candidate")
    control = _load_verified_control_artifacts(slot)
    enabled, proof_sha256 = _load_candidate_proof(candidate, slot, control)
    _runtime, prior = _load_latest_complete_attempt(
        candidate, slot, enabled.candidate_proof_id, proof_sha256
    )
    if prior.get("candidate_generated_fingerprint") != enabled.generated_fingerprint[:16]:
        raise evaluator.InfrastructureError(
            "candidate generated fingerprint does not match the prior evaluation"
        )
    if prior.get("control_generated_fingerprint") != control.generated_fingerprint[:16]:
        raise evaluator.InfrastructureError(
            "control generated fingerprint does not match the prior evaluation"
        )
    return control, enabled


@contextmanager
def acquire_existing_slot(
    source_repo: Path | None = None, slot_root: Path | None = None
) -> Iterator[Any]:
    """Lock an existing slot without invoking evaluator clone/build preparation."""

    source_input = (
        source_repo
        or Path(os.environ.get("GRHSIM_SOURCE_REPO", evaluator.DEFAULT_SOURCE_REPO))
    )
    source = source_input.expanduser().resolve()
    root_input = (
        slot_root or Path(os.environ.get("GRHSIM_SLOT_ROOT", evaluator.DEFAULT_SLOT_ROOT))
    ).expanduser()
    root_absolute = Path(os.path.abspath(root_input))
    current = Path(root_absolute.anchor)
    for component in root_absolute.parts[1:]:
        current /= component
        if current.is_symlink():
            raise evaluator.InfrastructureError(
                f"runtime-only slot path contains a symlink: {current}"
            )
    if not root_absolute.is_dir():
        raise evaluator.InfrastructureError(
            f"runtime-only slot root does not exist: {root_absolute}"
        )
    root = root_absolute.resolve(strict=True)
    namespace = hashlib.sha256(
        f"{source}\0{evaluator.PINNED_PARENT_COMMIT}\0{evaluator.PINNED_WOLVRIX_COMMIT}".encode(
            "utf-8"
        )
    ).hexdigest()[:16]
    namespace_path = root / namespace
    slot_path = namespace_path / "slot-0"
    for path, label in (
        (namespace_path, "namespace directory"),
        (slot_path, "slot directory"),
    ):
        if path.is_symlink() or not path.is_dir():
            raise evaluator.InfrastructureError(
                f"runtime-only retry requires an existing regular {label}: {path}"
            )
    try:
        slot_path.resolve(strict=True).relative_to(root)
    except (OSError, ValueError) as error:
        raise evaluator.InfrastructureError(
            f"runtime-only slot escapes its resolved root: {slot_path}"
        ) from error
    lock_path = slot_path / "lock"
    lock_file = evaluator._open_regular_lock(lock_path, create=False)
    timeout = float(os.environ.get("GRHSIM_SLOT_LOCK_TIMEOUT", "43200"))
    started = time.monotonic()
    acquired = False
    try:
        while True:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                if time.monotonic() - started >= timeout:
                    raise evaluator.InfrastructureError(
                        "timed out waiting for the existing GrhSIM evaluator slot"
                    )
                time.sleep(0.25)

        opened_lock = os.fstat(lock_file.fileno())
        try:
            current_lock = os.stat(lock_path, follow_symlinks=False)
        except OSError as error:
            raise evaluator.InfrastructureError(
                f"evaluator lock path changed after acquisition: {lock_path}: {error}"
            ) from error
        if (
            not stat.S_ISREG(current_lock.st_mode)
            or opened_lock.st_dev != current_lock.st_dev
            or opened_lock.st_ino != current_lock.st_ino
        ):
            raise evaluator.InfrastructureError(
                f"evaluator lock path was replaced after acquisition: {lock_path}"
            )

        control_repo = slot_path / "control"
        candidate_repo = slot_path / "candidate"
        results_dir = slot_path / "results"
        for path, label in (
            (control_repo, "control repository"),
            (candidate_repo, "candidate repository"),
            (results_dir, "results directory"),
        ):
            if path.is_symlink() or not path.is_dir():
                raise evaluator.InfrastructureError(
                    f"runtime-only retry requires an existing {label}: {path}"
                )
            try:
                path.resolve(strict=True).relative_to(slot_path.resolve(strict=True))
            except (OSError, ValueError) as error:
                raise evaluator.InfrastructureError(
                    f"runtime-only {label} escapes the slot: {path}"
                ) from error
        yield evaluator.Slot(
            slot_path, control_repo, candidate_repo, results_dir, lock_file
        )
    finally:
        try:
            if acquired:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("program_path")
    parser.add_argument("--source-repo", default=None)
    parser.add_argument("--slot-root", default=None)
    args = parser.parse_args()
    result = evaluator.evaluate(
        args.program_path,
        artifact_preparer=prepare_reused_artifacts,
        slot_acquirer=acquire_existing_slot,
        source_repo=args.source_repo,
        slot_root=args.slot_root,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
