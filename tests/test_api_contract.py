from apps.api.main import app


def test_openapi_exposes_recording_job_contract() -> None:
    paths = app.openapi()["paths"]

    assert "/v1/recordings" in paths
    assert "/v1/jobs/{job_id}" in paths
    assert "/v1/recordings/{recording_id}" in paths
    assert "/v1/recordings/{recording_id}/review-items" in paths
    assert paths["/v1/recordings"]["post"]["responses"]["202"]
