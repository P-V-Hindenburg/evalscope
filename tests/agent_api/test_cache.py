from evalscope.agent_api.cache import _cache_token


def test_cache_token_ignores_hub_separators():
    assert _cache_token('MVBench') == _cache_token('mv_bench')
    assert _cache_token('PKU-Alignment___mv_bench').endswith(_cache_token('MVBench'))
