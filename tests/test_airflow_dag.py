"""The Airflow DAG (pipelines/airflow/dags/deploy_model.py) against the pipeline.

The DAG runs in its own environment and never imports edgeinfer (D-038), so its stage list
is a copy: the first test checks the copy without Airflow. The second loads the DAG with
Airflow itself, in tools/airflow (marker `airflow`; a separate CI job)."""

import ast
import json
import os
import subprocess

import pytest

from edgeinfer.pipeline.runner import STAGES
from edgeinfer.reports import REPO_ROOT

DAG_FILE = REPO_ROOT / "pipelines" / "airflow" / "dags" / "deploy_model.py"


def test_dag_stages_are_the_pipeline_stages():
    tree = ast.parse(DAG_FILE.read_text())
    (stages,) = [ast.literal_eval(node.value) for node in tree.body
                 if isinstance(node, ast.Assign) and node.targets[0].id == "STAGES"]  # fmt: skip
    assert stages == STAGES


LOAD = """
import json
from airflow.dag_processing.dagbag import DagBag
bag = DagBag(dag_folder="pipelines/airflow/dags")
dag = bag.dags.get("deploy_model")
print(json.dumps({
    "errors": {k: str(v) for k, v in bag.import_errors.items()},
    "tasks": [t.task_id for t in dag.tasks] if dag else None,
    "upstream": [sorted(t.upstream_task_ids) for t in dag.tasks] if dag else None,
    "retries": sorted({t.retries for t in dag.tasks}) if dag else None,
    "commands": [t.bash_command for t in dag.tasks] if dag else None,
}))
"""


@pytest.mark.airflow
def test_dag_loads_in_airflow(tmp_path):
    env = os.environ | {"AIRFLOW_HOME": str(tmp_path), "AIRFLOW__CORE__LOAD_EXAMPLES": "False"}
    command = ["uv", "run", "--project", "tools/airflow", "python", "-c", LOAD]
    done = subprocess.run(command, cwd=REPO_ROOT, env=env, capture_output=True, text=True,
                          check=True)  # fmt: skip
    dag = json.loads(done.stdout.strip().splitlines()[-1])
    assert dag["errors"] == {}
    assert dag["tasks"] == list(STAGES)
    assert dag["upstream"] == [[]] + [[s] for s in STAGES[:-1]]  # a chain, in stage order
    assert dag["retries"] == [0]  # a failed gate is a verdict, not a transient error
    for stage, command in zip(STAGES, dag["commands"], strict=True):
        assert command.endswith(f"--stage {stage}") and "--run-id {{ dag_run" in command
