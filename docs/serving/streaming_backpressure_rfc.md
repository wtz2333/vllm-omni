# 流式视频背压 RFC

状态：实现对齐稿

范围：只覆盖 Roadmap Track B 中已经拆出的 PR 6（播放背压）和 PR 7（交互控制运行时）。camera/prompt 的具体语义另放在个人仓库的后续交互语义 PR 中。

## 1. 目标

流式扩散有两条需要分别验证的路径：

1. **播放路径**：客户端只接收并播放视频，不发送交互事件。引擎限制生成媒体领先播放游标的程度，并限制输出暂存和 inactive session 的资源占用。
2. **交互路径**：客户端在播放过程中发送 prompt/camera 等控制。控制进入独立的有界队列，在 engine-owned chunk boundary 应用；播放反馈、Stop 和 disconnect 继续走响应路径。

默认请求保持原来的 unpaced 行为。PR 6 和 PR 7 只服务单 stage、stepwise diffusion、`streaming_output=True`、`max_num_seqs=1` 的流式路径。

## 2. Roadmap 对应关系

| Roadmap Track B | PR 6 | PR 7 |
| --- | --- | --- |
| B1 pacing/output capacity | playback watermark、下一 chunk admission、host playback fast path | output credit、decoded/encoded byte limit、idle eviction、终态清理 |
| B2 deferred interaction boundary | 为交互保留 FIFO/RPC 隔离基础 | engine clock reservation、boundary context、resume 后应用 |
| B3 bounded resources/timing | playback/abort 响应路径 | bounded interaction/response queue、pacing metrics、consumer release |

本 RFC 不承诺 Track A cross-attention、TaoMate-H3 full request、NPU 优化或 AR experimental path 的行为。

## 3. 统一 API

公开请求使用一个 canonical pacing 对象：

```json
{
  "streaming_pacing": {
    "enabled": true,
    "mode": "client_playhead",
    "max_lead_ms": 1500,
    "max_output_bytes": 67108864,
    "idle_timeout_s": 30
  }
}
```

当前实现只接受 `client_playhead`。`session.playback.position_seconds` 是客户端累计播放游标；引擎会忽略回退值，并把未来值限制到已经生成的媒体长度。

PR 6 使用的 `streaming_buffer_seconds` 保留为兼容别名，并在入口处转换为 `max_lead_ms`。内部 runner 可以继续使用规范化后的秒数和资源字段，公共协议不再新增另一组同义配置名。

`received_at` 是 engine-owned 字段。客户端不能通过 `session.interaction` 或直接 `AsyncOmni` 调用注入时间戳。

## 4. Data flow

蓝线表示播放和输出路径，橙线表示交互控制路径。两条路径共享同一个 request state，但交互路径不会取代播放路径。

```mermaid
flowchart LR
    C[Client]
    W[Realtime WebSocket]
    A[AsyncOmni]
    O[Orchestrator]
    E[DiffusionEngine]
    R[Stepwise Runner]
    Q[Output queue / encoder]
    P[Playback cursor]
    I[Interaction queue]
    B[Engine boundary context]
    M[Chunk metadata]

    C -->|session.start| W
    W -->|request + streaming_pacing| A
    A -->|add request| O
    O -->|schedule before next chunk| E
    E -->|execute one chunk| R
    R -->|media output| Q
    Q -->|ordered media + metadata| C
    C -->|session.playback| P
    P -->|host fast path| E

    C -->|session.interaction| W
    W -->|reserve timestamp + bounded enqueue| I
    I -->|ordered interaction RPC| O
    O -->|track/release reservation| E
    E -->|boundary context| B
    B -->|eligible controls| R
    R -->|started/active/completed IDs| M
    M -->|chunk metadata| Q

    linkStyle 0,1,2,3,4,5,6,7,8,9 stroke:#2563eb,stroke-width:3px
    linkStyle 10,11,12,13,14,15,16,17 stroke:#f59e0b,stroke-width:3px
```

### 4.1 PR 6：无交互播放路径

```text
session.start(streaming_pacing)
  -> serving_video_output_stream
  -> AsyncOmni.generate
  -> Orchestrator.add_request
  -> DiffusionEngine._playback_blocked / scheduler.schedule
  -> execute_stepwise
  -> output queue
  -> encoder / WebSocket binary chunk
  -> client playhead feedback
  -> DiffusionEngine.update_streaming_playback
```

关键不变量：

- 下一 chunk admission 检查 `generated + next_chunk - played <= max_lead`。
- playback feedback 和 abort 不等待当前 GPU chunk 的普通 RPC。
- paced stream 保留一个 output credit，直到下游 consumer 继续 async iteration 或显式 release。
- Stop、disconnect、idle timeout 都能在没有下一次 inference 的情况下完成清理。

### 4.2 PR 7：交互路径

```text
session.interaction
  -> WebSocket control reader
  -> bounded interaction queue
  -> AsyncOmni.reserve_streaming_interaction
  -> engine monotonic timestamp
  -> ordered interaction RPC
  -> DiffusionEngine pending boundary
  -> InteractionBoundaryContext
  -> runner / modality handler
  -> output chunk metadata
```

PR 7 的通用运行时只负责：

- 接收时间戳并在 engine clock 中保存；
- 在播放暂停期间继续接收控制和取消；
- 在 resume boundary 把上下文传给 runner；
- 限制 transport queue、engine pending reservation 和 output retention；
- 报告 pacing 和 interaction 的可观测 metadata。

prompt transition、camera target/velocity、SE(3) 积分、coalescing 和暂停期间事件回放属于后续交互语义 PR。它们通过 `InteractionBoundaryContext` 接入，不改变 PR 6 的播放路径。

## 5. 状态与生命周期

| 状态 | 所有者 | 释放条件 |
| --- | --- | --- |
| `generated_seconds` / `played_seconds` | DiffusionEngine | request 完成、abort 或 idle eviction |
| output credit / retained bytes | DiffusionEngine + async consumer | consumer resume、显式 release 或 request abort |
| interaction reservation | DiffusionEngine | ordered RPC 完成、投递失败或取消 |
| boundary context | StepRequestState | 当前 chunk 应用后覆盖；request 清理时释放 |
| transport interaction queue | WebSocket session | delivery、disconnect 或 queue overflow |

每条完成、取消、断连和错误路径只能释放一次 request state。cleanup 不应依赖新的 inference wave。

## 6. 验收证据

| 类别 | PR 6 必须证明 | PR 7 必须证明 |
| --- | --- | --- |
| CPU contract | admission、playback fast path、abort、RPC teardown | boundary context、reservation、output credit、queue bounds |
| API | canonical `streaming_pacing`、兼容 alias、非法值 | interaction queued/failed、engine-only timestamp |
| GPU single rank | paced/unpaced 固定 seed 输出、lead 上限、Stop/idle 后新请求 | blocked send 时 control 响应、短交互到达时间、长序列无 retention 增长 |
| distributed | 多 rank boundary 一致、取消后所有 rank 清理 | 多 rank interaction/abort、断连和 final chunk 清理 |
| 资源 | decoded/encoded output bytes、峰值 retained bytes | interaction queue 和 response queue 在上限处有明确失败语义 |

当前远程 PR 的 CPU/GPU 结果属于功能证据；在 CI、multi-rank cancellation 和完整 serving-path 验收完成前，两个 PR 保持 draft。

## 7. 分支和 PR 关系

```text
PR 6: codex/diffusion-streaming-backpressure
  └── PR 7: codex/diffusion-interaction-timing
        └── follow-up: camera/prompt interaction semantics
```

PR 7 的最小生产范围是通用 interaction/backpressure runtime。camera/prompt 语义单独挂在个人仓库，后续根据模型和 Roadmap 验收证据决定是否推进到上游。
