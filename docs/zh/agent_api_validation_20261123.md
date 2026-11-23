# Agent HTTP API 最终验证记录（2026-11-23）

## 1. 验证范围

本次验证覆盖 HTTP 鉴权、Benchmark 查询、预检、异步提交、状态轮询、取消、结果返回、
多 Benchmark 调度、按 Benchmark 覆盖数量/子集、Judge、代码沙箱、多数据源路由、部分成功和超时清理。

## 2. 自动回归

- Agent API 及相关 EvalScope 原生配置测试：112/112 通过。
- `compileall` 和 `git diff --check` 通过。
- 改动仅位于 `evalscope/agent_api` 、`tests/agent_api` 和文档，未修改 EvalScope 原有评测核心。

## 3. Benchmark 实跑结果

| Benchmark | 模型 / Judge | 本轮规模 | 结果 | 主指标 |
| --- | --- | ---: | --- | --- |
| GSM8K | 本地 Qwen3-VL | 20 | 20/20 | Accuracy 0.25 |
| HellaSwag | 本地 Qwen3-VL | 20 | 20/20 | Accuracy 0.80 |
| IFEval | 本地 Qwen3-VL | 20 | 20/20 | Prompt strict 0.60 |
| LongBench-v2 | 豆包 | 4（short/medium） | 4/4 | Accuracy 0.75 |
| MT-Bench | 豆包 / DeepSeek | 20 | 20/20 | Judge 9.15 |
| HumanEval | 豆包 / 代码沙箱 | 20 | 20/20 | Pass@1 1.00 |
| POPE | 本地 Qwen3-VL | 20 | 20/20 | F1 0.80 |
| MMBench | 本地 Qwen3-VL | 20 | 20/20 | Accuracy 0.10 |
| DocVQA | 豆包 | 20 | 20/20 | ANLS 0.9375 |
| MIA-Bench | 豆包 / DeepSeek | 20 | 20/20 | Judge 0.9525 |
| MMMU | 豆包 | 4（4 学科） | 4/4 | Accuracy 1.00 |
| MVBench | 豆包 | 20 | 20/20 | Accuracy 0.65 |

注：上表为接口和兼容性验证，样本量不足以作为正式模型能力结论。

## 4. 最终混合任务

一次 HTTP 任务提交 GSM8K、POPE、MT-Bench、HumanEval 和 MVBench，共 17 条，
按数据源和数量自动拆分为 3 个执行组。

- 结果：17/17 成功，一份合并结果，无错误。
- 进度：从开始到结束 `total` 恒为 17；组进度按 0/3、1/3、2/3、3/3 递增。
- 报告阶段最高显示 99.9%，仅在终态进入 100%。
- Hugging Face 无法连网时，MVBench 预检正确识别本地缓存并继续运行。

## 5. 发现的问题与处理

### 5.1 多执行组总进度跳变

- 现象：旧逻辑只汇总已生成的 `progress.json`，进入新组时 `total` 才增长。
- 处理：任务开始时计算稳定计划量，并独立返回执行组进度。
- 结果：80 条及 17 条混合任务均未再出现总数跳变。

### 5.2 HTTP 后台线程 `fork` 子进程偶发死锁

- 现象：评测子进程在进入 EvalScope 前休眠。
- 根因：多线程 FastAPI 进程中 `fork` 会继承已锁互斥量。
- 处理：改用 `multiprocessing` 的 `spawn` 上下文启动干净子进程。
- 结果：后续 80 条、12 个 Benchmark 和最终混合任务均正常启动。

### 5.3 本地 Qwen 长上下文/大图片 OOM

- 现象：LongBench-v2 20/20 样本失败；DocVQA 4/20、MMMU 18/20 成功。
- 根因：GPU 0 同时有其他用户进程占用约 15.5 GiB，Qwen 服务明确返回 CUDA OOM。
- 处理：不终止其他用户进程；用豆包复测。DocVQA 20/20 通过，LongBench-v2 和 MMMU 缩小闭环通过。
- 结论：属于被测模型服务资源问题，不是统一 API 数据或调度错误；`partial_success` 返回正常。

### 5.4 慢 Benchmark 超过统一任务时限

- 现象：LongBench-v2 和 MMMU 的豆包 20 条复测在 30 分钟到期时未全部完成。
- 处理：保留超时记录，缩小任务确认链路；正式长任务应为不同 Benchmark 配置更长总时限。
- 修复：`evaluation exceeded ... seconds` 现在归类为 `TASK_TIMEOUT / timeout / retryable=true`，不再是 `unknown`。

### 5.5 SSH 断开

- 现象：长时间轮询期间 SSH 多次断开。
- 结果：FastAPI 后台任务继续运行，重连后可用原 `job_id` 恢复查询；MT-Bench、MIA-Bench 和 MVBench 均如此完成。
- 结论：异步提交/查询设计符合跨服务接入需求。

### 5.6 旧版 EvalScope 阶段精度

当前环境的 `progress.json` 不总是包含层级阶段，API 使用最新日志事件兼容回退。
样本进度、Benchmark 和生成/评分总体状态可查，但原生评测将预测与 Judge 按样本原子执行时，
不能始终单独展示一段持续的 `running_judge`。最终 Judge 覆盖率和错误仍从报告准确返回。

## 6. 结论

统一 HTTP API 已通过文本、图片、视频、Judge、沙箱、多数据源、不同子集/数量和多执行组合并验证，
可供上层 LangGraph Agent 进入联调。评测分数仅用于证明返回链路，正式评估仍需使用完整数据和独立资源。
