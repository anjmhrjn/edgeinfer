"""Gated deployment pipeline (C6): recipe -> artifacts -> gates -> registry.

The pipeline runs the existing CLI commands as stages and reads their reports; it never
imports an orchestrator (D-038). Airflow wraps it from `pipelines/airflow/dags/`.
"""
