# verl 仓库速览

## 这是什么

`verl` 是一个面向 LLM / VLM 后训练的强化学习框架。

它的核心定位不是单一算法实现，而是一个可组合的训练框架，重点解决这几件事：

- 用统一控制流组织 RL 后训练
- 把训练后端和推理后端解耦
- 支持单机、多机、多卡扩展
- 支持不同算法、不同 rollout 引擎、不同 reward 形式

一句话理解：

`verl = 单控制器 RL 编排层 + 分布式 worker 执行层 + 可插拔训练/推理/奖励后端`

## 这个仓库大体能做什么

从当前仓库内容看，它至少覆盖这些场景：

- LLM RL 后训练
  - PPO
  - GRPO
  - GSPO
  - REINFORCE++
  - ReMax
  - RLOO
  - 以及一批 recipe 里的扩展算法
- SFT
- 多轮 agent / tool calling RL
- 多模态 RL
  - Qwen2.5-VL
  - Qwen3-VL
  - Kimi-VL 等相关路径
- 函数式 reward
  - 数学、代码、规则验证类任务
- 模型式 reward
- 多机 Ray 集群训练
- 不同训练与 rollout 组合
  - FSDP / FSDP2 / Megatron-LM
  - vLLM / SGLang / HF rollout
- checkpoint 保存、恢复、合并为 Hugging Face 模型
- 评测、离线结果汇总、性能调优

如果从“用途”来分，这个仓库最像 4 个东西的组合：

1. 一个 RL post-training 框架
2. 一个分布式训练编排层
3. 一个算法实验平台
4. 一组可直接复用的 recipe / example

## 仓库结构怎么搭起来的

### 1. 核心代码在 `verl/`

这是框架本体。

- `verl/trainer/`
  - 训练入口和主控制流
  - 最重要的入口通常是 `verl/trainer/main_ppo.py`
- `verl/workers/`
  - 真正执行训练、推理、打分的分布式 worker
  - 包括 FSDP worker、Megatron worker、rollout worker 等
- `verl/single_controller/`
  - driver 如何通过 Ray 管 worker group
  - 是整个“单控制器编排”设计的关键
- `verl/protocol.py`
  - 定义 `DataProto`，是 batch 在各组件之间流动的统一协议
- `verl/utils/`
  - 数据集、配置、checkpoint、指标、并行、工具函数
- `verl/models/`
  - 模型适配和一些模型实现补丁
- `verl/experimental/`
  - 新特性和新范式
  - 比如 agent loop、reward loop、fully async、VLA 等
- `verl/interactions/`、`verl/tools/`
  - 多轮交互、工具调用相关逻辑

### 2. 可直接运行的样例在 `examples/`

这是最适合上手的目录。

按用途分得比较清楚：

- `examples/ppo_trainer/`
- `examples/grpo_trainer/`
- `examples/remax_trainer/`
- `examples/rloo_trainer/`
- `examples/reinforce_plus_plus_trainer/`
- `examples/sft/`
- `examples/sglang_multiturn/`
- `examples/data_preprocess/`
- `examples/slurm/`
- `examples/skypilot/`
- `examples/tutorial/`

经验上：

- 想跑通一个最小训练，先看 `examples/`
- 想理解“某个 feature 怎么配”，优先搜对应 example，而不是先钻核心代码

### 3. 算法 recipe 在 `recipe/`

这个目录更像“完整实验配方”而不是最小样例。

里面有很多更偏论文/项目复现的内容，例如：

- `dapo`
- `prime`
- `spo`
- `sppo`
- `retool`
- `r1`
- `entropy`
- `flowrl`
- `specRL`

可以把它理解成：

- `examples/` 偏基础模板
- `recipe/` 偏完整方案和专题实现

### 4. 文档在 `docs/`

如果你不是改底层，而是想快速会用，这里价值很高。

建议优先看：

- `docs/start/quickstart.rst`
- `docs/start/multinode.rst`
- `docs/examples/ppo_code_architecture.rst`
- `docs/hybrid_flow.rst`
- `docs/workers/`
- `docs/perf/`

### 5. 数据、评测、脚本是外围配套

- `dataset/`
  - 本地数据、评测集、训练集样例
- `evaluation/`、`eval/`
  - 评测脚本、结果汇总
- `scripts/`
  - 环境安装、模型合并、诊断脚本等
- `docker/`
  - 各类镜像和硬件环境构建方案
- `tests/`
  - 单测、分布式测试、特性测试

## 框架是怎么分层的

从设计上，这个仓库大概分 5 层：

### 1. 启动层

典型入口：

- shell 脚本
- `ray job submit`
- `python -m verl.trainer.main_ppo`

负责把任务启动起来。

### 2. 控制流层

典型文件：

- `verl/trainer/main_ppo.py`
- `verl/trainer/ppo/ray_trainer.py`

负责组织训练流程，例如：

- 读数据
- rollout
- reward
- advantage
- actor/critic update
- validate / save

### 3. 执行层

典型文件：

- `verl/workers/fsdp_workers.py`
- `verl/workers/megatron_workers.py`
- `verl/workers/rollout/...`

负责真正干活：

- 模型前向
- 生成
- logprob
- 参数更新

### 4. 协议层

典型文件：

- `verl/protocol.py`

负责统一 driver 和 worker 之间传的数据格式。

### 5. 配置层

典型文件：

- `verl/trainer/config/`

负责把算法、数据、actor、rollout、reward、trainer 等配置拼成一套完整运行参数。

## 对使用最有帮助的几个入口

### 如果你想“先跑起来”

先看：

- `README.md`
- `docs/start/quickstart.rst`
- `examples/ppo_trainer/`
- `examples/grpo_trainer/`

### 如果你想“看主执行逻辑”

先看：

- `verl/trainer/main_ppo.py`
- `verl/trainer/ppo/ray_trainer.py`
- `verl/protocol.py`

### 如果你想“看 worker 怎么干活”

先看：

- `verl/workers/fsdp_workers.py`
- `verl/workers/rollout/vllm_rollout/`
- `verl/single_controller/ray/base.py`

### 如果你想“换算法或改实验方案”

先看：

- `examples/*_trainer/`
- `recipe/`
- `verl/trainer/config/`

### 如果你想“做 agent / tool use / 多轮”

先看：

- `examples/sglang_multiturn/`
- `verl/experimental/agent_loop/`
- `verl/interactions/`
- `verl/tools/`

## 这个仓库的使用方式，建议这样理解

不要把它当成“某个脚本集合”，更合适的理解是：

- `examples/` 提供启动模板
- `config/` 决定行为
- `trainer/` 组织控制流
- `workers/` 执行计算
- `protocol.py` 负责传 batch
- `recipe/` 是专题实验工程

也就是说，平时真正需要改的通常不是很多：

- 换数据：改 `data.*`
- 换模型：改 `actor_rollout_ref.model.path`
- 换算法：改 `algorithm.*` 或换 example / recipe
- 换 rollout 后端：改 `actor_rollout_ref.rollout.name`
- 换并行策略：改 actor/critic/rollout 配置
- 换 reward：改 reward manager / reward loop / 自定义 reward function

## 几个对使用很有帮助的实际建议

### 1. 从 example 启动，不要从底层文件自己拼第一版命令

这个仓库配置项很多，直接裸写 `python -m ...` 很容易漏配置。

### 2. 先分清你是在用“框架能力”还是“recipe 能力”

- 只想训练一个自己的模型：优先 `examples/`
- 想复现实验或论文设定：优先 `recipe/`

### 3. 优先确认 4 个核心维度

每次读一个脚本，先看这 4 个点：

- 算法：PPO / GRPO / 其他
- 训练后端：FSDP / FSDP2 / Megatron
- rollout 后端：vLLM / SGLang / HF
- reward 形式：函数式 / 模型式 / reward loop

大部分执行逻辑分支，都是从这 4 个维度展开的。

### 4. 多机训练本质上是 Ray 作业

所以排查问题时，除了看训练脚本，也要看：

- Ray cluster 是否正常
- head / worker 是否都加入集群
- dashboard / job log 是否正常

### 5. `outputs/`、`result/`、`tensorboard_log/` 更像运行产物，不是核心源码

读仓库时不要被这些目录分散注意力。

### 6. 当前仓库里 `recipe/` 内容很多，适合抄结构，不适合一上来全读

更有效的方式是：

- 先选一个与你目标最接近的 recipe
- 再反向追它依赖的 trainer / worker / config

## 一个简短的目录地图

```text
verl/
  verl/                    # 框架本体
    trainer/               # 训练入口与控制流
    workers/               # 分布式执行单元
    single_controller/     # driver 与 worker group 编排
    protocol.py            # 统一 batch 协议
    utils/                 # 数据、配置、checkpoint、工具
    experimental/          # 新特性与实验模块
    models/                # 模型适配与补丁
  examples/                # 最适合上手的训练脚本模板
  recipe/                  # 更完整的专题方案/论文复现
  docs/                    # 官方文档
  dataset/                 # 本地数据与样例数据
  evaluation/, eval/       # 评测与结果分析
  scripts/                 # 安装、合并、诊断脚本
  docker/                  # 环境与镜像
  tests/                   # 测试
```

## 最后一句

如果你后面主要目标是“高效使用这个仓库”，最重要的不是把所有代码都看完，而是先建立这张映射：

`example / recipe -> config -> trainer -> worker -> rollout / reward`

把这条链路吃透之后，这个仓库就会从“很大很杂”变成“结构稳定、可替换组件很多”的框架。
