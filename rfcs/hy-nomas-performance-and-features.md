# RFC：Hy-Nomas 持续生成、性能优化与特性接入

状态：提案，尚未实施。日期：2026-10-26。本文是个人仓库中的中文讨论稿，供后续拆分实现 PR 使用。

## 1. 动机与目标

PR [#9338](https://github.com/vllm-project/vllm-omni/pull/9338) 已完成 `tencent/Hy-Nomas-Diffusers` 的基础接入，包括文本和相机条件生成、多轮续生成、稳定帧流式输出、Ulysses SP2 和模块级 CPU offload。后续需要解决三个实际问题：

1. 持续生成受到固定历史容量限制；当前达到上限后需要结束或重置会话。
2. 已有性能测量出现基线漂移，且缺少流式首帧、逐 chunk 延迟及阶段分析，尚不能据此选择最有效的优化。
3. 上游的历史选择、KV 压缩、解码加速和部分交互能力尚未通过 Omni 的统一接口提供。

本 RFC 提议保留现有 BF16 正确性基线，先建立可信测量，再补齐长期运行所需的状态管理，随后根据实测瓶颈推进性能优化和扩展特性。所有改变历史上下文、权重精度或 attention 计算的方案都单独配置、单独验收。

第一阶段以一至两张 H200 为主要验证设备；更大并行度需要独立验证。本文不承诺未经测量的加速倍数，也不以其他硬件上的上游结果替代本地结果。

## 2. 版本与已有证据

| 对象 | 固定版本 |
| --- | --- |
| 本 RFC 分支起点 | vLLM-Omni `ad3a730e870cdaf4d4856d08b34052159e772725`，编写时拉取的 upstream/main |
| 已验证的 Hy-Nomas 接入 | `257856d3afce694803f83fd60674f292b7a34583`，PR #9338；不假定已进入 main |
| 模型组件与参考实现 | Tencent-Hunyuan/Hy-Nomas `911cea6458a89c2663513ff7099561ec190dd491` |
| BF16 组件权重 | tencent/Hy-Nomas-Diffusers `a1ae04d11ae4ea176d1f76de634c115944f9882d` |
| 独立原生权重参考 | tencent/Hy-Nomas `bac719a80c8d02f91c485fca804bb8221e149b70` |
| 已发布测试材料 | 个人仓库证据提交 `a518affb28c71bf072bae2e4397dd5f227822e75`，见文末链接 |
| 编写时核对的上游 main | `df76395ddf93fcfc9edb4816f130a3ccebdbd45a`；相比固定参考版本增加了 Reactor 服务及演示等内容 |

后续实现从当时最新 main 开发，依赖 PR #9338 的部分在其合入或明确叠加后执行。若依赖版本改变，先重跑受影响的基线；不得把旧版本结果归到新版本。

已有实验使用 H200（每卡 141 GB），Python 3.12.13、PyTorch 2.13.0 / CUDA 13.0、vLLM 0.32.0、Transformers 5.16.1、Diffusers 0.40.0、FlashAttention 2.8.3，驱动 570.133.20。后续实验另行记录实际环境。

### 2.1 正确性与功能证据

- 435 个 CPU 测试通过，4 个未选中；这些结果包含替代学习组件的测试，不能代替真实模型验证。
- 18 个真实 GPU 测试通过，覆盖精度、单卡和 SP2 的离线与视频 API，以及 resident / CPU offload。
- 六个普通生成对照：文本、分段提示词、相机控制，各覆盖 resident / offload；384×640、49 帧、8 steps、seed 42，全部 decoded float pixels 的 MAE、MSE、max abs 均为 0，PSNR 为无穷大，逐帧 SSIM 为 1。
- 两个续生成精度对照：每个包含两轮 49 帧，验证每轮 latent 和完整 98 帧输出一致；另有 SP2 三轮 147 帧的稳定输出验证。
- SP2 流式验收包含一次 477 帧交互生成、一次产生 23 个解码帧后的取消，以及同一服务上的新请求完成 477 帧。

这些结果证明对应样例的正确性与生命周期行为；尚未证明长时间运行质量、流式延迟、并发吞吐或新增特性的正确性。视频链接及原始样本保存在文末的固定版本测试材料中。

### 2.2 已有性能结果

负载为一个固定 forest prompt，704×1280、121 帧、8 steps、BF16、seed 42、CFG=1、flow shift=5、batch/concurrency=1。同一对通过 NVLink 连接的 H200，每项 2 次同形状 warmup、5 次测量。SP1 使用其中第一张卡。

计时范围：从 `Omni.generate` 调用到完整 CPU tensor 返回，包含文本编码、DiT/KV、PixelUMM、格式转换和传输；不包含模型加载、显存计数 RPC、哈希、视频编码与写盘。以下是普通离线生成数据，不能用作 AR 流式性能基线。

| 配置 | 中位 E2E（ms） | 均值 ± 标准差（ms） | 每张卡峰值 allocated / reserved（GiB） |
| --- | --- | --- | --- |
| SP1，resident | 30894.7 | 32602.0 ± 7422.3 | GPU 0：59.09 / 63.37 |
| SP2，resident | 21123.7 | 21173.7 ± 198.2 | GPU 0：53.05 / 57.06；GPU 1：53.05 / 57.06 |
| SP1，module offload | 41248.6 | 41361.2 ± 3601.0 | GPU 0：30.56 / 32.07 |
| SP2，module offload | 38466.6 | 39800.6 ± 2961.5 | GPU 0：23.80 / 25.99；GPU 1：23.80 / 25.99 |
| SP1，resident，末尾复测 | 25669.2 | 25693.3 ± 78.6 | GPU 0：59.09 / 63.37 |

末尾 SP1 中位数比起始 SP1 低 16.91%。因此，原表中的 1.463× SP2 比值属于探索结果，不能作为稳定加速结论。五个样本的 nearest-rank p95 就是最大值，不足以说明尾部延迟。

SP2 的均值对应 5.715 个生成帧/秒，低于输出视频的 24 fps；这是串行离线生成速率，不能据此宣称服务达到实时播放速度。当前还没有足够数据判断 DiT、KV、解码或传输各自占比。

## 3. 当前执行和状态边界

| 入口 | 当前执行方式 | 输出与状态 |
| --- | --- | --- |
| 普通离线与 `/v1/videos` | 普通 diffusion 路径，pipeline 内执行生成循环 | 每个提示词段作为一个降噪块，最终整段解码；KV 只在本次请求内使用 |
| 多轮续生成 | `ARDiffusionEngine`，`step_execution=false` | 一个 tick 完成一个 chunk，跨 tick 保留 KV、噪声与解码状态 |
| `/v1/realtime/video` | `ARDiffusionEngine`，`step_execution=true` | 在 chunk 边界应用提示词/相机更新，增量解码并发送稳定帧 |

AR 路径使用 `ARDiffusionSegmentedKVState` 保存 BF16 历史。每个 chunk 完成降噪后执行缓存写入 forward，保留全部旧 handles 和新 handles。该 forward 保留上游 `cache_noise_sigma` 和时间步语义，不能当成可以删除的重复降噪。

当前属于全历史保留，具有固定容量上限；`window_frames=1` 在 segmented 路径中不构成滑窗策略。默认容量为 120 个 latent 帧、16 个文本段、一个活跃会话。单次连续请求最多约 477 帧；多轮输出按各轮实际帧数累计，不能直接使用一个总 latent 数推算。

`ar_max_latent_frames` 同时影响容量和预采样噪声时间线。单纯提高它会增加显存，并可能改变相同 seed 的噪声结果。会话关闭、取消、失败和淘汰会释放状态。正常完成发送解码尾帧；取消丢弃尚未发送的尾帧。

已有 segmented staging 按 owner/revision 复用同一历史，避免在同一个 chunk 的每一步重新收集历史。后续优化应针对新提交后的收集、缓冲空间或真正分页读取，不重复实现已有复用。

## 4. 特性范围与接入顺序

| 能力 | 当前接入 | 后续范围与优先级 |
| --- | --- | --- |
| 统一 AR 状态管理 | 流式和续生成使用 AR；普通生成仍有独立 KV 路径 | P1：统一状态所有权，保留各入口原有采样和输出语义 |
| 长历史策略 | 固定上限内保留全部历史 | P1：先接最近历史窗口和按段淘汰；P2：接入上游历史采样策略 |
| KV 存储 | GPU BF16 | P1：评估按阶段 CPU 卸载；P2：INT8，随后单独评估 QVG |
| 权重卸载 | module offload | P1：优先检查分步执行中的驻留区间和可暴露的传输成本 |
| 解码与文本编码 | 各 SP rank 复制组件 | P1：根据阶段占比选择解码并行或文本编码器分片 |
| Attention / compile | dense FlexAttention，eager | P1：等价后端与受限编译；P2：稀疏 attention |
| 量化权重与 LoRA | BF16 distilled，无 LoRA | P2：INT8 DiT、FP8 PixelUMM；P3：官方适配器及合并权重接入 |
| 会话交互 | 提示词/相机更新、续生成、取消、重置 | P2：撤销上一段和暂停恢复，配套协议与客户端行为 |
| 提示词增强 | 普通文本或已准备的结构化提示词 | P3：可选 Nomas-PE 服务接入，分别计时和处理服务失败 |
| 完整浏览器交互 | 共享流式客户端 | P3：复用现有媒体接口接入按键/触控相机与界面，不引入第二套推理服务 |

上游原生配置包含 `recent`、`uniform`、`recent_mix_uniform`、`temporal_context` 等历史策略。`recent` 使用视频和文本共享的 token 预算，配置的历史帧数不保证全部用于视频帧。采样策略可以保留 CPU 历史归档，因此“GPU 历史有界”不代表主机历史也有界。QVG、滑窗、稀疏 attention、撤销等能力存在组合限制，必须按上游代码检查，不能直接排列组合。

## 5. 设计

### 5.1 共用执行接口，保留入口语义

以现有 AR capability、runner 和 tick/interaction 协议为基础，让引擎统一拥有会话和 KV，模型提供几何信息、条件计算和状态提交规则。参考模型的 DiT、PixelUMM 与 Euler 数学计算继续使用现有固定版本；逐步替换组件需要独立的精度证明。

普通生成迁入 AR 状态管理时，仍保持原来的“一提示词段一降噪块”和最终 float video 输出。流式继续使用 chunk 调度和稳定帧输出。不能直接把普通请求改为固定 16-latent chunk 后宣称生成行为不变，也不能把普通入口原先接受的长度隐式限制为流式默认容量。

普通请求按已知总长度进行容量准入；持续会话按明确的历史、噪声和解码预算准入。兼容验证覆盖 API 默认值、帧数/时长、seed、generator、分段边界、相机条件以及输出 layout/dtype。

### 5.2 解耦时间线长度与驻留历史

分别管理累计生成位置、GPU 保留历史、CPU 归档和当前未提交 chunk。全历史模式继续作为默认参考；窗口与采样模式由使用者显式选择。

第一项历史策略实现复用上游 `recent` 或按段淘汰的语义，不自行发明不同的 sink/prefix 保留规则。进入窗口以后，输出相对全历史参考可能改变，属于上下文变化，必须与启用相同策略的上游比较。

历史选择同时作用于视频 KV、所需文本、可见性 mask 和 action-tail 范围。RoPE 时间位置、累计帧位置和相机坐标保留原始时间线含义，不能随物理存储压缩而重新从零编号。保留集合只能在所有层和所有 rank 成功提交后改变；失败时保留旧的已提交状态。

持续生成还必须处理固定噪声库和 PixelUMM 时间线，不能只增加 KV 淘汰。噪声扩展遵循上游增量规则，保留 RNG 状态、draw shape 与顺序；先验证旧容量内结果，再验证越过原 120-latent 边界。若新噪声协议无法保持旧 seed 的结果，显式区分生成版本，不静默改变旧模式。

阶段验收是连续生成至少 120 秒，跨越多次窗口淘汰和提示词更新；GPU 驻留历史在配置预算内稳定。CPU 归档、噪声和解码历史分别设定上限，达到上限返回明确行为。采样归档与完整历史模式不承诺无限时长。

### 5.3 优化 KV 读取和存储

首先测量新 chunk 开始时的历史收集、clean-cache 写入和 staging 成本。已有分页存储不代表 attention kernel 已经直接读取分页 KV；当前参考 attention 使用连续 views。

按风险分两步：

1. 在保留集合是旧集合加新段时复用已有连续历史，仅追加新的有效区间；owner、revision 或保留次序改变时重新收集。验证跨会话切换、淘汰、reset 和失败后不会读到旧数据。
2. 只有连续 staging 的时间或显存成为主要成本时，评估直接分页 attention。必须保留视频/文本联合 attention、action mask、RoPE 与 SP 语义；通用 paged attention 接口存在不代表可以直接替换。

KV 体积的估算使用 `2 × layers × local_kv_heads × head_dim × bytes_per_element × retained_tokens`。连续 staging、页面取整、当前 chunk、噪声和解码状态另算；不得只按 KV 页体积报告总显存节省。

### 5.4 以完整 chunk 为单位管理组件驻留

普通生成已经在整个 chunk 的降噪及缓存提交外保持 DiT 驻留。分步流式执行中的多次 forward 可能重复搬运 DiT，是否构成主要瓶颈需要 trace 证明；已有普通生成 offload 表不能证明流式搬运占比。

拟让共享 offloader 在 chunk 开始后持有 DiT，完成全部 denoising steps 和缓存写入后释放，再进入 PixelUMM。取消、异常和 rank 失败均必须结束驻留。KV CPU offload 作为独立选项处理，不把权重卸载描述为 KV 卸载。

先比较文本编码器卸载、DiT 卸载、全组件卸载，报告首 chunk、稳定阶段和每张卡峰值显存。预取和重叠仅在暴露的传输时间足够大时引入，并核对额外缓冲是否破坏容量收益。

### 5.5 优化解码和并行布局

根据 trace 确认 PixelUMM 在关键路径中的比例后，比较当前逐 rank 复制解码、指定 rank 解码和上游时间窗口并行。指定 rank 解码需要检查现有输出归属、collective 顺序及其他 rank 的错误通知，不能只跳过非主 rank 的调用。

流式解码必须保持稳定前缀和 overlap/tail 规则。正常 finish 完整发送尾帧；取消不补发尾帧。对比相同 latent 的解码输出，覆盖短 chunk 无稳定帧、多个窗口、最后一块、失败后新请求。

文本编码器分片主要作为显存优化。Ulysses SP1/SP2 先建立稳定性能基线，再按可用设备评估 SP4/SP8；分辨率 token 数、head 数和拓扑必须满足约束。CFG=1 的当前模型不因支持 CFG parallel 就自动获得收益。多请求吞吐优先比较独立副本，连续批处理另设容量与尾延迟目标。

### 5.6 量化、编译和近似计算

分别接入 INT8 KV、INT8 DiT 和 FP8 PixelUMM，不在首个实验中同时改变三者。BF16 attention 配合 INT8 存储主要节省容量，不预设 GEMM 或 attention 加速；记录反量化、scale、临时缓冲和传输开销。FP8 解码 kernel 按 H200 的实际支持验证，不复用面向其他 GPU 的性能声明。

编译先选择固定 shape 的局部计算区域，显式记录编译次数、首次开销、缓存命中和额外显存。KV 地址、历史长度、mask 和控制更新是独立的动态因素。只有语义和地址稳定后才评估 CUDA Graph，不将 Python 状态改变放入 capture。

MagiAttention 等后端切换验证计算差异；稀疏 attention、历史采样和减少采样步数进入质量预算路线。当前只有 4/6/8 steps，跳过一步的相对影响较大，Cache-DiT/TeaCache 不作为默认优先项。任何跳步方案都要报告实际执行的 DiT forward 次数，并保留必要的缓存写入 forward。

### 5.7 补齐交互能力

撤销上一段需要同时恢复 KV 保留集合及量化 scales、mask、累计位置、prompt/camera 状态、RNG 和 PixelUMM 状态。先支持上游允许撤销的历史策略；纯 recent 模式没有完整归档时明确拒绝。

已经发送的追加式视频不能原地撤回。撤销需要客户端协商时间线版本和替换范围，再重新生成该范围；已有只接受追加视频的客户端保持原协议。取消仍释放会话，不把取消隐式改成暂停。暂停恢复需限定保留时长与显存预算，并覆盖断开连接和服务终止。

Nomas-PE 作为可选的前置服务，将其耗时与视频生成耗时分别报告。保留直接提交结构化提示词的入口；增强服务失败返回明确错误，不默默改用另一种输入语义。相机键盘/触控输入转成已有相机控制，不另建模型执行通道。

### 5.8 模块归属

| 模块 | 拟承担的修改 |
| --- | --- |
| `vllm_omni/diffusion/models/hy_nomas/` | 声明历史选择和模型几何、保持采样/条件/解码语义、处理模型状态；不另建分页分配器 |
| `vllm_omni/experimental/ar_diffusion/` | 会话生命周期、segmented KV 的存储和原子提交、所需通用状态接口；不加入 Hy-Nomas 专属算法 |
| `vllm_omni/diffusion/offloader/` | 组件驻留生命周期；能力复用现有 component discovery 与卸载配置 |
| 现有 interaction、视频入口和共享客户端 | 控制事件、时间线修订、暂停恢复、背压和资源释放；保持媒体传输归属清楚 |
| Hy-Nomas CPU / accuracy / offline / online tests，以及共享回归 | 覆盖模型数值、状态边界和实际服务行为；复用当前 helpers 与 CI 标记 |

涉及共享接口的 PR 同时验证现有使用方。只有模型需要的配置留在模型目录，避免为了一个模型增加通用配置层。

## 6. 性能候选与收益边界

证据等级：A 为同模型、同硬件、稳定重复 A/B 且具有阶段归因；B 为同模型的受限实验；C 为源码或上游机制支持、尚缺本地测量。当前没有足够证据给候选优化填写可靠的预期毫秒收益。

下表的 `T_x` 指 profiler 确认的关键路径时间，不是重叠 annotation 或所有 kernel 时间之和；`O` 是新增成本。收益为 baseline 减 candidate，负值表示变慢。预期收益在 P0 完成后填写；乐观上界不能作为承诺。

| 优先级 / 证据 | 候选与已有依据 | 影响阶段 / 乐观上限 | 保守 / 预期 / 乐观 E2E 节省 | 实现与风险 | 显存、吞吐与依赖 |
| --- | --- | --- | --- | --- | --- |
| P0 / B | 重建 SP/offload 基线；已有 16.91% 漂移 | 测量与归因 | 不计算加速收益 | 固定设备与交错顺序，低风险 | 为全部性能声明提供前提 |
| P1 / C | 流式 DiT 驻留；代码显示逐 forward 的组件上下文 | 暴露传输 `T_transfer` | `−O / 待测 / T_transfer` | 共享 offloader 的 chunk 生命周期；中风险 | 峰值可能增加；依赖流式 trace |
| P1 / C | KV 增量 staging；已有同 revision 复用 | 历史收集 `T_stage` | `−O / 待测 / T_stage` | append 条件和失效规则；中风险 | 降低拷贝，缓冲显存未必降低 |
| P1 / C | 解码布局；当前各 rank 复制 PixelUMM | 关键路径解码 `T_decode` | `−O / 待测 / T_decode` | 单 rank 或窗口并行；中高风险 | 可减少冗余，新增通信；需同 latent 精度通过 |
| P1 / C | 文本编码器分片 / KV 阶段卸载 | 权重与 KV 容量 | `−O / 待测 / T_encode 或 T_transfer` | 复用共享管理，分开实验；中高风险 | 主要目标是容量；可能降低单请求速度 |
| P1 / C | 局部编译 / 等价 attention 后端 | 可归因的算子或 launch 时间 `T_kernel` | `−O / 待测 / T_kernel` | 固定区域和失效规则；中高风险 | 记录编译与额外缓存，依赖动态状态稳定 |
| P2 / C | INT8 KV / INT8 DiT / FP8 解码 | 分别测量对应容量和计算阶段 | `−O / 待测 / T_component` | 标准量化接口和严格加载；高风险 | 不按压缩比推断加速；需各组件质量验证 |
| P2 / C | 历史选择 / 稀疏 attention | 历史 attention `T_attention` | `−O / 待测 / T_attention` | 保留上游策略语义；高风险 | 可限制 GPU 历史，CPU 归档另计；需长程质量预算 |

每个候选的正式实验使用第 7 节的同输入 A/B，并使用第 8 节的验收规则。实施拆分和前置条件见第 9 节。局部算子提升的端到端估算采用：

`saving_ms = T_stage × (1 − 1 / stage_speedup) − added_overhead_ms`

多个作用于同一阶段的收益不能相加。每接受一个优化，重新测量基线后再估算下一项。无法测到收益或收益落在噪声范围内的方案不升级为默认配置。

## 7. 实验协议

### 7.1 固定工作负载

| 编号 | 负载 | 比较目的 |
| --- | --- | --- |
| W1 | 384×640、49 帧、8 steps；文本 / 两段提示词 / 相机；seed 42 | 复用已有独立参考精度样例 |
| W2 | 704×1280、121 帧、8 steps、seed 42；固定 forest prompt | 复测已有 SP1/SP2 与 offload 数据 |
| W3 | 704×1280；三轮各 49 帧，最后 close；resident / offload | 续生成状态、稳定帧和最终解码 |
| W4 | 704×1280、477 帧；在固定 chunk 更新 prompt 和 camera；取消后新请求 | 流式首帧、逐 chunk、控制延迟和资源恢复 |
| W5 | 704×1280、至少 120 秒；窗口预算 18/36/72 latent 历史帧分别实验 | 跨越原容量上限、多个淘汰周期和长程质量 |
| W6 | 12 个固定场景 × 3 个 seeds；8 steps 主比较，4/6 steps 各做补充 | 量化、稀疏、采样历史的质量与控制一致性 |

W2 使用已有原始 prompt：`A forest beside a clear river, photorealistic, daylight, gentle camera motion.`。W3/W4 复用已发布样例并冻结具体控制事件；执行前把所有 prompt、seed、事件 chunk、相机文件及哈希写入 manifest。W6 覆盖主体保持、风格变化、运动、场景切换、空间记忆、连续相机及文本/相机交替；不能事后选择更好看的样例替代。

### 7.2 测量方法

- 所有 A/B 固定代码与依赖版本、权重、输入、seed、分辨率、帧数、步数、精度、并行布局和设备互连，记录唯一变化。普通分段与流式 chunk 不能混为同一 A/B。
- 首轮每项 2 次同形状 warmup、5 次测量用于检查协议。正式入选候选采用三轮 A-B-B-A，每块先 warmup，再保留 5 次有效请求，每种至少 30 个样本；输入顺序一致。相邻基线块的中位数漂移超过 5% 时排查并重测。
- 报告样本数、中位数、均值、标准差、原始样本和 p95；给出按块比较的置信区间。小样本 p95 明确标注其局限。加速比为 baseline / candidate，时延下降为 `1 − candidate / baseline`。
- profiler 只用于阶段归因；正式时延在无 profiler 的独立测量中报告。GPU 异步执行使用正确的完成边界，多 rank 取实际关键路径，不加总重叠阶段。
- 为文本编码、每个 DiT step、缓存写入、历史收集、解码、H2D/D2H、编码和发送增加必要观测；记录 NCCL 暴露时间、rank 等待及 CPU/GPU 空闲区间。
- 首可播放帧从客户端发送请求计时，到客户端成功解码第一帧结束；另外记录首稳定帧、首媒体包、逐 chunk 时间、解码帧数、最终尾帧和更新生效延迟，不能以首包代替首帧。
- 分别报告纯生成 E2E、含媒体编码的服务 E2E 和客户端可播放时间。串行生成 fps、输出视频 fps、RTF、并发吞吐分别计算；RTF 为 wall time / 输出媒体时长。
- 显存表逐卡列出 PyTorch peak allocated / reserved，所有 rank 在计时外收集；按相同生命周期重置计数。需要驱动层占用时另列，并说明采样间隔。公开表格不混入主机内存数值；主机归档是否超预算在运行记录中单独检查。

### 7.3 复用入口与材料

普通生成复用 `examples/offline_inference/text_to_video/text_to_video.py` 和 `hy_nomas.yaml`；续生成复用 `hy_nomas_session.py`；流式复用 `hy_nomas_ar.yaml` 与共享 streaming client。代码路径以 PR #9338 的固定提交为依据，在后续 main 上重新核对。

已有回归入口如下，必须先准备固定模型、参考依赖与匹配设备数：

```bash
python -m pytest tests/diffusion/models/hy_nomas -m cpu --run-level core_model -q
python -m pytest tests/e2e/accuracy/test_hy_nomas.py \
  -m 'local_model and cuda' --run-level core_model -q
python -m pytest tests/e2e/offline_inference/test_hy_nomas.py \
  tests/e2e/online_serving/test_hy_nomas.py \
  -m 'local_model and cuda and cards_2' --run-level core_model -q
```

历史 benchmark 的实际参数、原始 JSONL 和脚本使用文末证据 README。未来实现新增的命令必须随对应 PR 提供，本文不把尚未存在的 benchmark CLI 写成可运行命令。实验媒体与一次性分析保存在代码仓库外，公开材料提供固定版本链接。

## 8. 正确性、质量与运行验收

| 改变类型 | 必需验收 |
| --- | --- |
| 状态管理、驻留和纯拷贝优化 | 保持相同数学路径；W1/W3 的 latent 与 decoded float 输出继续满足原来的精确断言；覆盖所有 rank 的资源释放 |
| 更换 attention / 编译 kernel | 先定位第一处数值差异；与同输入参考逐层和完整输出比较；容差在正式实验前按算法和 BF16 误差确定并记录，不能因测试失败临时放宽 |
| 窗口和采样历史 | 与相同策略、相同预算的上游比较；另外与全历史版本做 W5/W6 质量比较；跨越淘汰边界，检查 mask、RoPE 和相机时间连续性 |
| KV / 权重量化、稀疏计算 | W6 全样本保留；报告 latent 误差、decoded MAE/PSNR/SSIM、逐帧变化和完整视频；数值指标不能单独证明长期质量 |
| 交互、撤销和暂停恢复 | 验证事件确认和实际生效 chunk；撤销后状态及重生成与对应参考一致；检查断开、超时、取消及后续新请求 |
| 组合与长时间运行 | 先分别验证再组合；至少 120 秒生成和 30 分钟重复会话/取消循环，记录泄漏趋势、失败、队列和事件积压 |

W6 的质量预算需要在执行前固定：建议把预定义控制/时序用例通过率下降不超过 2 个百分点作为讨论起点，同时进行隐藏配置标签的完整视频对照，逐项记录主体漂移、相机连续性和闪烁。该数值是待确认的验收提案，不是已达到的模型指标；质量预算未确认前保留为实验配置，不能设为默认。

窗口内短视频通过不代替 W5。撤销测试至少覆盖普通、INT8 与允许撤销的采样策略；不能把未提交 KV 的异常 rollback 当成已经支持用户撤销。长流输出还要覆盖客户端读取变慢、队列有界、尾帧处理和所有 rank 一致终止。

## 9. 拆分为可独立验收的 PR

| 顺序 | 交付内容 | 前置条件 | 完成标准 |
| --- | --- | --- | --- |
| P0 | 稳定 benchmark、流式指标、阶段 trace、原始材料 | 基础接入可用 | W2/W4 有可信基线；漂移检查通过；没有把估算写成实测 |
| P1-A | AR 状态所有权统一及普通入口兼容 | 基础接入 | W1/W3 旧断言通过，普通分段和输出语义保留 |
| P1-B | 历史保留策略与噪声/解码时间线解耦 | P1-A，固定上游策略 | W5 跨越 120-latent 上限及多次淘汰，资源预算有效 |
| P1-C | chunk 驻留与 KV 增量 staging，分别提交 | P0 的传输/收集证据 | W1/W3/W4 通过，每项独立 A/B 证明价值 |
| P1-D | 解码布局或文本编码器分片 | P0 的阶段/显存证据 | 同 latent 解码验证通过，逐卡显存与 E2E 同时报告 |
| P2-A | INT8 KV，随后独立接入 INT8 DiT / FP8 PixelUMM | 相应基线和质量预算 | W5/W6 与组合限制清楚；加载、scale 和失败路径有覆盖 |
| P1-E / P2-B | P1 的局部编译/等价后端，以及 P2 的历史采样/稀疏优化，按机制拆分 | P0；相关状态接口稳定 | 按机制通过第 8 节，测到有意义收益 |
| P2-C | 撤销、暂停恢复与客户端协议 | 历史与 decoder 状态可恢复 | 状态恢复和客户端时间线一致，旧客户端行为明确 |
| P3 | Nomas-PE、LoRA、浏览器控制 | 相应模型/服务与共享协议 | 单独可关闭、可复现，异常路径与实际效果通过验证 |

每个 PR 同时提交相关回归测试、适用的本地检查、真实 GPU 验证范围和结果。测试进入 CI 时按当前 L1/L2/L3/L4 规则选择层级；手工大矩阵不全部转成 nightly。

## 10. 待讨论的决策与退出标准

需要讨论的是默认是否继续保留全历史、长期运行的 GPU/CPU 预算、窗口与量化的质量预算，以及撤销所需的客户端协议。提案默认保持当前模式，扩展能力使用显式配置；实现前确认会影响兼容性的选择。

保持计算语义的优化路线：完成 P0 后，以同一工作负载 E2E 中位数下降至少 10%，或每张卡峰值 allocated 下降至少 15% 作为候选价值标准，同时通过正确性、尾延迟和生命周期检查。这些是选择优化的目标，不是已经测得的收益。

质量预算路线：在明确历史、主机归档和解码预算下连续生成至少 120 秒，通过 W5/W6 和已确认的质量标准，再根据实测选择窗口、量化或稀疏配置。无法满足预算、收益或正确性标准的方案保留为实验结果，不替换现有默认配置。

## 11. 参考资料

- [基础接入 PR #9338](https://github.com/vllm-project/vllm-omni/pull/9338)
- [固定版本的接入实现与 recipe](https://github.com/wtz2333/vllm-omni/blob/257856d3afce694803f83fd60674f292b7a34583/recipes/Tencent/Hy-Nomas.md)
- [原始指标、全部比较视频、流式片段及复现脚本](https://github.com/wtz2333/vllm-omni/blob/a518affb28c71bf072bae2e4397dd5f227822e75/test-evidence/PR9338/hy-nomas-diffusers/README.md)
- [Hy-Nomas-Diffusers 模型与组件说明](https://huggingface.co/tencent/Hy-Nomas-Diffusers)
- [上游固定版本的历史、量化和编译配置](https://github.com/Tencent-Hunyuan/Hy-Nomas/blob/911cea6458a89c2663513ff7099561ec190dd491/configs/config.yaml)
- [上游低显存配置](https://github.com/Tencent-Hunyuan/Hy-Nomas/blob/911cea6458a89c2663513ff7099561ec190dd491/configs/experiment/single_gpu_low_mem_inference.yaml)
- [上游会话、解码与撤销说明](https://github.com/Tencent-Hunyuan/Hy-Nomas/blob/911cea6458a89c2663513ff7099561ec190dd491/apps/README.md)
- [Reactor 流式交互说明，核对版本 df76395](https://github.com/Tencent-Hunyuan/Hy-Nomas/blob/df76395ddf93fcfc9edb4816f130a3ccebdbd45a/reactor/README.md)

AI assistance：Codex 辅助核对代码和已有测试材料，并起草本 RFC。本文中的新方案尚未实现，新增实验尚未运行，人工审阅待完成。
