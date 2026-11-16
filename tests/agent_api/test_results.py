from evalscope.agent_api.results import artifact_refs


def test_artifact_refs_preserve_evalscope_native_layout():
    refs = artifact_refs('/tmp/run-1')

    assert refs.predictions == '/tmp/run-1/predictions'
    assert refs.reviews == '/tmp/run-1/reviews'
    assert refs.reports == '/tmp/run-1/reports'
