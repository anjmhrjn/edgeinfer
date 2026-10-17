import pytest

from edgeinfer.pipeline import registry
from edgeinfer.pipeline.registry import RegistryError, load, publish, rollback


def _record(n, target="trt"):
    return {"model": "m", "artifact_id": f"id{n}", "sha256": f"sha{n}", "precision": "fp16",
            "target": target, "location": f"instance:artifacts/engines/m/e{n}.engine",
            "recipe": "m", "recipe_hash": "h", "run_id": f"run{n}", "deploy_report": "d",
            "git_sha": "abc", "gate_reports": {}}  # fmt: skip


def _files(*present):
    """A sha lookup where versions in `present` exist with their published sha256."""
    shas = {f"instance:artifacts/engines/m/e{n}.engine": f"sha{n}" for n in present}
    return shas.get


def test_publish_registers_versions_and_moves_the_alias(tmp_path):
    assert publish(_record(1), "production", tmp_path) == (1, True)
    assert publish(_record(2), "production", tmp_path) == (2, True)
    index = load("m", tmp_path)
    assert index["aliases"] == {"production": 2}
    assert [(h["action"], h["from"], h["to"]) for h in index["history"]] == [
        ("publish", None, 1), ("publish", 1, 2)]  # fmt: skip


def test_publishing_the_same_artifact_keeps_its_version(tmp_path):
    publish(_record(1), "dev", tmp_path)
    assert publish(_record(1), "dev", tmp_path) == (1, False)
    index = load("m", tmp_path)
    assert len(index["versions"]) == 1 and len(index["history"]) == 1  # no alias move


def test_same_artifact_on_another_target_is_a_new_version(tmp_path):
    publish(_record(1), "dev", tmp_path)
    assert publish(_record(1, target="ort_cpu"), "dev", tmp_path) == (2, True)


def test_unknown_alias_is_refused(tmp_path):
    with pytest.raises(RegistryError, match="alias"):
        publish(_record(1), "prod", tmp_path)


def test_rollback_goes_back_one_version_and_not_to_a_rolled_back_one(tmp_path):
    for n in (1, 2, 3):
        publish(_record(n), "production", tmp_path)
    assert rollback("m", "production", _files(1, 2, 3), "bad v3", tmp_path) == (3, 2)
    assert rollback("m", "production", _files(1, 2, 3), "bad v2", tmp_path) == (2, 1)
    with pytest.raises(RegistryError) as e:
        rollback("m", "production", _files(1, 2, 3), "again", tmp_path)
    assert e.value.code == "REGISTRY_NO_PREVIOUS"
    index = load("m", tmp_path)
    assert index["aliases"]["production"] == 1
    assert index["history"][-1] == {**index["history"][-1], "action": "rollback", "from": 2,
                                    "to": 1, "reason": "bad v2"}  # fmt: skip


def test_republishing_a_rolled_back_version_makes_it_a_target_again(tmp_path):
    publish(_record(1), "production", tmp_path)
    publish(_record(2), "production", tmp_path)
    rollback("m", "production", _files(1, 2), "bad v2", tmp_path)  # -> v1
    publish(_record(2), "production", tmp_path)  # fixed elsewhere, promoted on purpose
    publish(_record(3), "production", tmp_path)
    assert rollback("m", "production", _files(1, 2, 3), "bad v3", tmp_path) == (3, 2)


def test_rollback_checks_the_artifact_is_still_there(tmp_path):
    publish(_record(1), "production", tmp_path)
    publish(_record(2), "production", tmp_path)
    with pytest.raises(RegistryError, match="doesn't exist") as e:
        rollback("m", "production", _files(2), "bad v2", tmp_path)  # v1's engine is gone
    assert e.value.code == "REGISTRY_ARTIFACT_MISSING"

    def rebuilt(location):  # same path, new bytes: a rebuild must pass the gates again
        return "other" if location.endswith("e1.engine") else None

    with pytest.raises(RegistryError, match="replaced"):
        rollback("m", "production", rebuilt, "bad v2", tmp_path)
    assert load("m", tmp_path)["aliases"]["production"] == 2  # nothing moved


def test_rollback_without_an_alias(tmp_path):
    with pytest.raises(RegistryError, match="points at no version"):
        rollback("m", "production", _files(), "x", tmp_path)


def test_local_sha_reads_files_and_refuses_instance_locations(tmp_path):
    f = tmp_path / "model.onnx"
    f.write_bytes(b"onnx")
    assert registry.local_sha(str(f)) == registry.local_sha(str(f)) is not None
    assert registry.local_sha(str(tmp_path / "missing.onnx")) is None
    with pytest.raises(RegistryError):
        registry.local_sha("instance:artifacts/engines/m/e1.engine")
