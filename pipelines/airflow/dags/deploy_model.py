"""Airflow DAG: one deploy run of a recipe, one task per pipeline stage (D-038).

Each task is `edgeinfer deploy run --stage <stage>` for the same run id, so Airflow
orders, logs and shows the stages, while what each stage does stays in the pipeline (and
in the reports it writes). State passes between tasks through the run's deploy report and
the content-addressed artifacts, not XCom. Airflow runs in its own environment
(tools/airflow, D-038) and never imports edgeinfer: the tasks call its CLI.

A task fails when its stage exits non-zero: 1 when a gate or a producing stage blocks the
deployment, 2 when the stage couldn't run. There are no retries: a failed gate is a
verdict, and a rerun of the DAG reuses every stage that already passed (D-040).

Run locally:
    AIRFLOW_HOME=artifacts/airflow AIRFLOW__CORE__DAGS_FOLDER=pipelines/airflow/dags \
    AIRFLOW__CORE__LOAD_EXAMPLES=False uv run --project tools/airflow \
    airflow dags test deploy_model -c '{"recipe": "deeplabv3plus_mbv2__ort_cpu"}'
"""

from __future__ import annotations

from pathlib import Path

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG, Param

REPO = Path(__file__).resolve().parents[3]

# Must equal edgeinfer.pipeline.runner.STAGES (checked by tests/test_airflow_dag.py).
STAGES = ("export", "convert", "validate", "push", "build", "parity", "accuracy", "perf",
          "publish")  # fmt: skip

# One run id for every task of a DAG run, in the pipeline's own format (UTC, sortable).
RUN_ID = "{{ dag_run.start_date.strftime('%Y%m%dT%H%M%SZ') }}"

with DAG(
    dag_id="deploy_model",
    description="Gated deployment of one recipe (configs/deploy/)",
    schedule=None,
    catchup=False,
    params={
        "recipe": Param(
            "deeplabv3plus_mbv2__ort_cpu",
            type="string",
            description="configs/deploy/<name>.yaml, or a path",
        )
    },
    default_args={"retries": 0},
    tags=["edgeinfer"],
) as dag:
    previous = None
    for stage in STAGES:
        task = BashOperator(
            task_id=stage,
            bash_command=(
                f"cd {REPO} && uv run --no-sync edgeinfer deploy run "
                "--recipe {{ params.recipe }} "
                f"--run-id {RUN_ID} --stage {stage}"
            ),
            do_xcom_push=False,
        )
        if previous is not None:
            previous >> task
        previous = task
