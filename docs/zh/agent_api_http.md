# EvalScope Agent HTTP API

本文说明如何将 EvalScope 的 LLM 和 VLM 统一评测能力作为独立 HTTP 服务接入上层 LangGraph Agent。
服务只负责评测执行、状态和结果，不负责理解用户自然语言或选择 Benchmark。

## 1 架构与调用流程

```text
LangGraph Agent
        ↓ HTTP
EvalScope Agent FastAPI
        ↓ Python API
TaskConfig + run_task
        ↓
Benchmark、评分与报告
```

推荐使用异步流程：

```text
提交评测 → 获得 job_id → 查询状态 → 获取结果
```

## 2 安装

在 EvalScope 项目目录和隔离环境中安装：

```bash
pip install -e '.[agent_api]'
```

## 3 环境变量

真实密钥只保存在 EvalScope 服务端。调用方只能传递逻辑引用，例如 `deepseek` 或 `doubao`。

```bash
export DEEPSEEK_API_KEY='<server-side-secret>'
export ARK_API_KEY='<server-side-secret>'
export EVALSCOPE_SERVICE_TOKEN='<shared-service-token>'
export EVALSCOPE_ALLOWED_ENDPOINT_HOSTS='api.deepseek.com,ark.cn-beijing.volces.com'
```

可使用以下命令生成服务 Token：

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

默认凭据映射为：

| `credential_ref` | 服务端环境变量 |
|---|---|
| `deepseek` | `DEEPSEEK_API_KEY` |
| `doubao` | `ARK_API_KEY` |

如需增加其他模型，可配置 JSON 映射：

```bash
export EVALSCOPE_CREDENTIAL_MAP='{"qwen":"QWEN_API_KEY"}'
```

禁止在请求、代码、文档或 Git 中写入真实模型密钥。

## 4 启动服务

```bash
uvicorn evalscope.agent_api.service.app:app \
  --host 0.0.0.0 \
  --port 8001 \
  --workers 1
```

第一版任务状态保存在当前 Python 进程中，因此必须使用一个 worker。服务重启后内存任务状态会丢失，
但 EvalScope 已生成的输出文件仍保留。生产环境后续应由平台或数据库持久化任务状态。

调用方自己的等待时限与服务端任务时限是两回事。如果调用方不再等待，必须调用取消接口；
服务端收到取消请求后会终止 EvalScope 工作进程，若普通终止信号无效，会在清理时限后强制结束进程。

启动后可访问：

```text
GET  http://<evalscope-host>:8001/health
Docs http://<evalscope-host>:8001/docs
```

除 `/health` 和文档页面外，业务请求必须携带：

```text
Authorization: Bearer <EVALSCOPE_SERVICE_TOKEN>
```

## 5 接口列表

| 方法与路径 | 作用 |
|---|---|
| `GET /health` | 检查服务是否运行 |
| `GET /v1/benchmarks` | 查询 LLM/VLM Benchmark |
| `GET /v1/benchmarks/{benchmark_id}` | 查询单个 Benchmark 详情 |
| `POST /v1/evaluations/preflight` | 评测前检查，不执行付费推理 |
| `POST /v1/evaluations` | 异步提交评测 |
| `GET /v1/evaluations/{job_id}` | 查询状态和进度 |
| `GET /v1/evaluations/{job_id}/result` | 获取统一评测结果 |
| `DELETE /v1/evaluations/{job_id}` | 请求取消任务 |

## 6 查询 Benchmark

```bash
curl -H "Authorization: Bearer $EVALSCOPE_SERVICE_TOKEN" \
  'http://<evalscope-host>:8001/v1/benchmarks?capability=math&modality=text&verified_only=true'
```

查询单项：

```bash
curl -H "Authorization: Bearer $EVALSCOPE_SERVICE_TOKEN" \
  http://<evalscope-host>:8001/v1/benchmarks/gsm8k
```

单项详情中的 `subsets` 字段列出该 Benchmark 当前可选择的子集。提交评测时可以通过
`benchmark_options.<benchmark_id>.subsets` 选择其中一个或多个子集。

## 7 运行前检查

```bash
curl -X POST http://<evalscope-host>:8001/v1/evaluations/preflight \
  -H "Authorization: Bearer $EVALSCOPE_SERVICE_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "target": {
      "model_id": "deepseek-chat",
      "api_url": "https://api.deepseek.com/v1",
      "credential_ref": "deepseek",
      "eval_type": "openai_api",
      "modality": "text"
    },
    "benchmarks": ["gsm8k"],
    "limit_per_subset": 3
  }'
```

`ok=true` 表示可以进入正式提交阶段。预检会检查 Benchmark、模态、Judge、凭据、依赖、磁盘、
模型端点和数据源，但不会调用模型生成答案。

## 8 提交评测

```bash
curl -X POST http://<evalscope-host>:8001/v1/evaluations \
  -H "Authorization: Bearer $EVALSCOPE_SERVICE_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "target": {
      "model_id": "deepseek-chat",
      "api_url": "https://api.deepseek.com/v1",
      "credential_ref": "deepseek",
      "eval_type": "openai_api",
      "modality": "text"
    },
    "benchmarks": ["gsm8k"],
    "limit_per_subset": 3,
    "generation": {
      "temperature": 0.0,
      "timeout": 60,
      "retries": 1,
      "retry_interval": 0
    }
  }'
```

成功时返回 HTTP 202：

```json
{
  "job_id": "eval_...",
  "status": "queued"
}
```

HTTP 请求不能指定 `work_dir`、`use_cache` 或服务器本地文件路径。输出目录由服务自动管理。

`generation.timeout` 是单次模型调用尝试的超时时间，`generation.retries` 是总尝试次数，
`generation.retry_interval` 是失败后再次尝试前的等待秒数。因此一次样本调用的最长等待时间
大约为 `timeout × retries`，另加少量连接和清理开销。对响应较慢或成本敏感的服务，建议从
`timeout=60, retries=1` 开始，再按实际稳定性增加重试。

### 8.1 全局每子集数量

`limit_per_subset` 的含义是“每个被选中子集最多运行多少条”，不是整个 Benchmark 的总样本数。
该语义与 EvalScope 原生 `TaskConfig.limit` 一致。

```json
{
  "benchmarks": ["gsm8k", "pope", "mmmu"],
  "limit_per_subset": 20
}
```

如果 POPE 选择了3个子集，则每个子集最多20条；MMMU 同理。正式全量评测可以不传该字段。

### 8.2 单 Benchmark 覆盖

`benchmark_options` 可以为个别 Benchmark 覆盖全局数量，并选择子集：

```json
{
  "benchmarks": ["gsm8k", "pope", "mmmu"],
  "limit_per_subset": 20,
  "benchmark_options": {
    "pope": {
      "limit_per_subset": 10,
      "subsets": ["popular", "random"]
    },
    "mmmu": {
      "limit_per_subset": 5,
      "subsets": ["Accounting", "Biology"]
    }
  }
}
```

最终配置为：GSM8K 使用全局值20；POPE 的 `popular`、`random` 各最多10条；MMMU 的
`Accounting`、`Biology` 各最多5条。单项配置优先于全局配置。

也可以不设置全局值，只为部分 Benchmark 单独设置数量；其他未设置数量的 Benchmark 将运行
所选子集的全部数据。

`benchmark_options` 的键必须同时出现在 `benchmarks` 中；`subsets` 不能留空或重复，且必须来自
Benchmark 查询接口返回的 `subsets`。HTTP 接口不开放底层 `dataset_args`，避免调用方传入服务器
本地路径或其他不安全配置。

### 8.3 为什么保留全局字段

现有调用方已经使用 `limit_per_subset`。本次增强不删除也不改变它的含义，只增加可选的
`benchmark_options` 覆盖层，因此原有请求、脚本和 LangGraph Tool 无需修改即可继续运行。
服务内部会按照数据源和最终 `limit_per_subset` 自动拆分执行组，最后仍返回一个 `job_id` 和一份
统一结果；这种分组不会限制不同 Benchmark 使用不同数量。

## 9 查询状态

```bash
curl -H "Authorization: Bearer $EVALSCOPE_SERVICE_TOKEN" \
  http://<evalscope-host>:8001/v1/evaluations/<job_id>
```

状态包括：

```text
queued
validating
loading_dataset
running_model
running_judge
scoring
generating_report
completed
partial_success
failed
cancelled
```

进度中同时返回样本和执行组两个层次：

```json
{
  "progress": {
    "current": 40,
    "total": 80,
    "percent": 50.0,
    "total_is_estimate": true,
    "current_benchmark": "gsm8k",
    "completed_groups": 1,
    "total_groups": 2,
    "current_group": 2,
    "current_group_benchmarks": ["gsm8k", "hellaswag"],
    "stage": "running_model"
  }
}
```

- `total` 在任务开始时尽可能根据整数 `limit_per_subset` 和子集计算，之后不会因为进入下一执行组而跳变。
- `total_is_estimate=true` 表示这是运行前上限；某些数据集的实际样本可能少于限制，任务结束时会用真实数量校准并变为 `false`。
- 全量或比例限制无法在运行前精确计算时，`total` 保持 `null`，直到所有执行组的数量都已知，避免将第一组误当成整个任务。
- `completed_groups/total_groups` 表示多数据源或多数量配置被拆成的执行组进度；它不改变最终仍返回一份合并结果的行为。

## 10 获取结果

```bash
curl -H "Authorization: Bearer $EVALSCOPE_SERVICE_TOKEN" \
  http://<evalscope-host>:8001/v1/evaluations/<job_id>/result
```

任务尚未完成时返回 HTTP 202。完成后返回 HTTP 200，主要字段包括：

```json
{
  "run_id": "eval_...",
  "status": "completed",
  "benchmarks": [
    {
      "benchmark_id": "gsm8k",
      "model_id": "deepseek-chat",
      "metrics": [
        {
          "name": "accuracy",
          "score": 1.0,
          "num": 3,
          "is_primary": true
        }
      ]
    }
  ],
  "sample_summary": {
    "planned": 3,
    "successful": 3,
    "failed": 0,
    "skipped": 0,
    "scored": 3,
    "error_counts": {}
  },
  "artifacts": [],
  "errors": []
}
```

`artifacts` 中的路径位于 EvalScope 服务器。跨服务器下载日志和报告的接口将在后续版本补充。

## 11 取消任务

```bash
curl -X DELETE \
  -H "Authorization: Bearer $EVALSCOPE_SERVICE_TOKEN" \
  http://<evalscope-host>:8001/v1/evaluations/<job_id>
```

返回 HTTP 202 表示取消请求已接受，不代表子进程已经立即退出。继续查询状态，直到变为 `cancelled`。

## 12 Python 调用示例

```python
import time

import httpx

base_url = 'http://<evalscope-host>:8001'
headers = {'Authorization': 'Bearer <service-token>'}
request = {
    'target': {
        'model_id': 'deepseek-chat',
        'api_url': 'https://api.deepseek.com/v1',
        'credential_ref': 'deepseek',
        'eval_type': 'openai_api',
        'modality': 'text',
    },
    'benchmarks': ['gsm8k'],
    'limit_per_subset': 3,
}

with httpx.Client(base_url=base_url, headers=headers, timeout=30.0) as client:
    response = client.post('/v1/evaluations', json=request)
    response.raise_for_status()
    job_id = response.json()['job_id']

    while True:
        status = client.get(f'/v1/evaluations/{job_id}').json()
        if status['status'] in {'completed', 'partial_success', 'failed', 'cancelled'}:
            break
        time.sleep(2)

    result = client.get(f'/v1/evaluations/{job_id}/result')
    result.raise_for_status()
    print(result.json())
```

## 13 豆包 VLM 配置

当前已验证的是 OpenAI Chat Completions 兼容方式，`api_url` 使用服务根地址，不直接填写完整
`/responses` 路径：

```json
{
  "model_id": "doubao-seed-evolving",
  "api_url": "https://ark.cn-beijing.volces.com/api/v3",
  "credential_ref": "doubao",
  "eval_type": "openai_api",
  "modality": "multimodal"
}
```

## 14 错误格式

HTTP 错误统一返回：

```json
{
  "error": {
    "code": "JOB_NOT_FOUND",
    "message": "unknown job_id: eval_xxx",
    "retryable": false
  }
}
```

常见错误包括鉴权失败、未知 Benchmark、未知凭据引用、服务端凭据缺失、模型地址未获准、
请求字段错误和未知任务。

## 15 当前限制

- 第一版必须使用一个 Uvicorn worker。
- 服务重启后内存中的 job 状态丢失，输出文件仍保留。
- 任务状态通过轮询获取，尚未主动推送事件。
- `artifacts` 是 EvalScope 服务器本地路径，尚未提供远程下载接口。
- 第一版不提供跨服务器任意缓存路径和输出路径。
- 小样本测试只用于链路验证，不能作为正式模型能力结论。
