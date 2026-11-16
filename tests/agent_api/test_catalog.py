from evalscope.agent_api import get_benchmark, list_benchmarks


def test_get_benchmark_exposes_runtime_requirements():
    humaneval = get_benchmark('humaneval')
    mt_bench = get_benchmark('mt_bench')
    mvbench = get_benchmark('mvbench')

    assert humaneval.requires_sandbox is True
    assert mt_bench.requires_judge is True
    assert mvbench.default_dataset_hub == 'huggingface'
    assert mvbench.modalities == ['text', 'video']


def test_list_benchmarks_filters_capability_and_verification():
    benchmarks = list_benchmarks(capability='math', modality='text', verified_only=True)
    ids = {item.benchmark_id for item in benchmarks}

    assert 'gsm8k' in ids
    assert all(item.verified for item in benchmarks)
