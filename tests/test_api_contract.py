from apps.api.main import app


def test_openapi_exposes_recording_job_contract() -> None:
    paths = app.openapi()["paths"]

    assert "/v1/recordings" in paths
    assert "/v1/jobs/{job_id}" in paths
    assert "/v1/jobs/{job_id}/trace" in paths
    assert "/v1/recordings/{recording_id}" in paths
    assert "/v1/recordings/{recording_id}/review-items" in paths
    assert "/v1/operations/release" in paths
    assert "/v1/operations/release/activate" in paths
    assert "/v1/operations/release/rollback" in paths
    assert "/v1/operations/retention" in paths
    assert "/v1/operations/retention/apply" in paths
    assert "/v1/families/{family_id}/replays" in paths
    assert paths["/v1/recordings"]["post"]["responses"]["202"]
