from pathlib import Path

import main_wizard
import pytest


def _write_task(root: Path, family: str, task: str, instruction: str | None) -> None:
    task_dir = root / family / task
    task_dir.mkdir(parents=True)
    (task_dir / "init_program.py").write_text("# EVOLVE-BLOCK-START\n# EVOLVE-BLOCK-END\n")
    (task_dir / "evaluator.py").write_text("def evaluate(path): return {'combined_score': 0}\n")
    if instruction is not None:
        (task_dir / instruction).write_text("task specification\n")


@pytest.mark.parametrize("extension", ["triton", "cuda"])
def test_discover_tasks_accepts_code_as_instruction(tmp_path, monkeypatch, extension):
    _write_task(tmp_path, "gpukernel", "trimul", f"trimul.{extension}")
    monkeypatch.setattr(main_wizard, "DATASETS_DIR", tmp_path)

    tasks = main_wizard.discover_tasks()

    assert [task.label for task in tasks] == ["gpukernel/trimul"]
    assert tasks[0].instruction == tmp_path / "gpukernel" / "trimul" / f"trimul.{extension}"


def test_discover_tasks_prefers_text_instruction(tmp_path, monkeypatch):
    _write_task(tmp_path, "demo", "example", "instructions.txt")
    task_dir = tmp_path / "demo" / "example"
    (task_dir / "example.triton").write_text("kernel specification\n")
    monkeypatch.setattr(main_wizard, "DATASETS_DIR", tmp_path)

    tasks = main_wizard.discover_tasks()

    assert tasks[0].instruction == task_dir / "instructions.txt"


def test_discover_tasks_skips_task_without_instruction(tmp_path, monkeypatch):
    _write_task(tmp_path, "demo", "incomplete", None)
    monkeypatch.setattr(main_wizard, "DATASETS_DIR", tmp_path)

    assert main_wizard.discover_tasks() == []


def test_repository_exposes_all_21_task_packages():
    tasks = main_wizard.discover_tasks()

    assert len(tasks) == 21
    assert {task.label for task in tasks} == {
        "ahc/ahc039",
        "ahc/ahc058",
        "autocorrelation/autocorrelation_first",
        "autocorrelation/autocorrelation_second",
        "autocorrelation/autocorrelation_third",
        "circle_packing/circle_packing_26",
        "circle_packing/circle_packing_32",
        "erdos/erdos_min_overlap",
        "gpukernel/asymmetricmatmul",
        "gpukernel/cumsum",
        "gpukernel/trimul",
        "hadamard_maximal_det/hadamard_maximal_det_29",
        "numerical_tasks/lasso_path",
        "open_problems_bio/denoising",
        "qubit_routing/swap_reduction",
        "scaling_law/domain_mixture_scaling_law",
        "scaling_law/easy_question_scaling_law",
        "scaling_law/lr_bsz_scaling_law",
        "scaling_law/parallel_scaling_law",
        "sums_diffs/sums_diffs",
        "znaa/znaa",
    }
