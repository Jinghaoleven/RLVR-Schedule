# verl 仓库执行逻辑说明

本文基于以下几个入口文件梳理：

- 启动脚本：`examples/grpo_trainer/run_qwen3-4b-curriculum-XCombined-multi-node.sh`
- 主入口：`verl/trainer/main_ppo.py`
- 训练主循环：`verl/trainer/ppo/ray_trainer.py`

目标不是泛泛介绍 verl，而是把你这个脚本在这个仓库里到底是怎么跑起来的、主要代码文件分别做什么、数据怎么流动，讲清楚。

## 1. 先给结论：这个脚本实际启动了什么

你的脚本本质上是在做一套基于 Ray 的多机 GRPO 训练，执行拓扑可以概括成：

1. shell 脚本负责启动多机 Ray 集群
2. head 节点通过 `ray job submit` 提交一个 Python 任务
3. Python 任务入口是 `python3 -m verl.trainer.main_ppo`
4. `main_ppo.py` 用 Hydra 读取默认配置 + shell 脚本里的覆盖参数
5. `TaskRunner` 在 Ray 上创建训练控制器
6. `RayPPOTrainer` 构建 worker group、数据集、dataloader，并进入训练循环
7. 每个 step 大体执行：
   - 从 parquet 读一批 prompt
   - rollout 生成响应
   - reward loop 计算奖励
   - actor 重新算 old log prob
   - driver 侧算 GRPO advantage
   - actor 更新参数
   - 按频率验证/保存 checkpoint

对你这个具体脚本，真正活跃的核心组件不是“PPO 全家桶”，而是下面这几个：

- `ActorRollout` 混合 worker：负责 actor 训练和 rollout 权重同步
- `vLLM rollout replica`：真正执行生成
- `AgentLoopManager`：把 batch 切给多个 agent loop worker 异步调度 rollout
- `RewardLoopManager`：用 reward worker 计算奖励
- `RayPPOTrainer`：单进程 driver，负责控制流

反而下面两个组件在你这个脚本里默认是不活跃的：

- `Critic`：你用的是 `algorithm.adv_estimator=grpo`，默认不启 critic
- `RefPolicy`：你设置了 `actor.use_kl_loss=False` 且 `algorithm.use_kl_in_reward=False`，默认不启 reference policy

所以，这个任务更准确地说是：

`多机 Ray + FSDP actor + vLLM rollout + reward loop + driver 侧 GRPO`

## 2. 当前脚本的执行链路

### 2.1 Shell 层

文件：`examples/grpo_trainer/run_qwen3-4b-curriculum-XCombined-multi-node.sh`

它做了 4 件事：

1. 设置环境变量
   - `WORKING_DIR`
   - `HYDRA_FULL_ERROR`
   - `HF_HUB_OFFLINE=1`
   - NCCL 相关配置

2. 依据外部注入的环境变量决定多机角色
   - `MASTER_ADDR`
   - `WORLD_SIZE`
   - `RANK`
   - `NPROC_PER_NODE`

3. 启动 Ray
   - `RANK=0` 节点执行 `ray start --head`
   - 其他节点执行 `ray start --address=...`

4. 只有 head 节点真正提交训练任务
   - `ray job submit --address=http://$MASTER_ADDR:8265 -- python3 -m verl.trainer.main_ppo ...`

所以这份 shell 脚本不是直接训练，它只是“拉起 Ray 集群 + 提交 Python 作业”。

### 2.2 Python 入口层

文件：`verl/trainer/main_ppo.py`

入口函数：`main()`

主要逻辑：

1. Hydra 读取 `verl/trainer/config/ppo_trainer.yaml`
2. 用 shell 覆盖参数合并配置
3. 调 `run_ppo(config)`
4. 如果当前进程还没 `ray.init()`，就先初始化 Ray runtime env
5. 创建一个远程 `TaskRunner`
6. 在 Ray 上执行 `runner.run.remote(config)`

注意：真正的训练控制逻辑不在 `main()` 里，而在 `TaskRunner.run()` 里。

### 2.3 TaskRunner 装配层

文件：`verl/trainer/main_ppo.py`

`TaskRunner.run()` 做的事情按顺序是：

1. 打印最终解析后的 config
2. 根据配置决定使用什么 worker
   - `add_actor_rollout_worker()`
   - `add_critic_worker()`
   - `add_ref_policy_worker()`
3. 校验配置 `validate_config(...)`
4. 把模型路径 copy 到本地可访问位置
5. 初始化 tokenizer / processor
6. 初始化 reward manager 或 reward loop
7. 创建 train / val dataset
8. 创建 sampler
9. 构造 `RayPPOTrainer`
10. `trainer.init_workers()`
11. `trainer.fit()`

这一步的本质是：把“配置”翻译成“运行时对象”。

## 3. 你这个脚本对应的有效运行拓扑

结合脚本参数，当前任务的有效分支如下。

### 3.1 算法分支

脚本里有：

- `algorithm.adv_estimator=grpo`
- `algorithm.use_kl_in_reward=False`
- `actor_rollout_ref.actor.use_kl_loss=False`

于是：

- advantage 用 `GRPO`
- 不需要 reference policy
- 默认不需要 critic

对应代码判断在：`verl/trainer/ppo/utils.py`

- `need_critic(config)`
- `need_reference_policy(config)`
- `need_reward_model(config)`

### 3.2 worker 分支

脚本没有显式改 `trainer.use_legacy_worker_impl`，默认是 `auto`。

同时：

- `actor.strategy` 默认是 FSDP 路线
- `rollout.name=vllm`

因此 actor/rollout worker 实际会落到：

- `verl/workers/fsdp_workers.py` 中的 `AsyncActorRolloutRefWorker`

这个 worker 同时承载：

- actor 训练
- rollout 权重切换
- 可选 reference policy

### 3.3 rollout 分支

脚本指定：

- `actor_rollout_ref.rollout.name=vllm`
- `actor_rollout_ref.rollout.tensor_model_parallel_size=2`
- `actor_rollout_ref.actor.ulysses_sequence_parallel_size=2`

因此 rollout 后端核心在：

- `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`
- `verl/experimental/agent_loop/agent_loop.py`
- `verl/experimental/agent_loop/single_turn_agent_loop.py`

### 3.4 reward 分支

默认配置：

- `reward_model.enable=False`
- `reward_model.use_reward_loop=True`

这意味着：

- 不启独立 reward model
- 但启 reward loop worker
- 奖励来自函数式 reward manager，而不是单独的 RM 模型

对应核心代码：

- `verl/trainer/ppo/reward.py`
- `verl/experimental/reward_loop/reward_loop.py`
- `verl/trainer/config/reward_manager.yaml`
- `verl/trainer/config/reward_model/reward_model.yaml`

## 4. 推荐你按这个顺序读代码

如果你想最短路径理解这个仓库，我建议按下面顺序读：

1. `examples/grpo_trainer/run_qwen3-4b-curriculum-XCombined-multi-node.sh`
2. `verl/trainer/main_ppo.py`
3. `verl/trainer/ppo/ray_trainer.py`
4. `verl/trainer/ppo/utils.py`
5. `verl/protocol.py`
6. `verl/utils/dataset/rlpro_dataset.py`
7. `verl/experimental/agent_loop/single_turn_agent_loop.py`
8. `verl/experimental/agent_loop/agent_loop.py`
9. `verl/workers/fsdp_workers.py`
10. `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`
11. `verl/experimental/reward_loop/reward_loop.py`
12. `verl/single_controller/ray/base.py`

这样读的原因是：先看控制流，再看数据协议，再看 rollout 和 worker 细节，最后再看 Ray 封装。

## 5. 主要代码文件及作用

| 文件 | 作用 |
|---|---|
| `examples/grpo_trainer/run_qwen3-4b-curriculum-XCombined-multi-node.sh` | 多机启动脚本，负责拉起 Ray 并提交训练任务 |
| `verl/trainer/main_ppo.py` | PPO/GRPO 主入口；做配置解析、TaskRunner 装配、dataset/sampler 构建 |
| `verl/trainer/config/ppo_trainer.yaml` | 训练总配置入口，聚合 actor/data/reward/rollout/model/critic 配置 |
| `verl/trainer/config/data/legacy_data.yaml` | 数据相关默认配置，包括 train/val 文件、batch size、prompt/response 长度等 |
| `verl/trainer/config/rollout/rollout.yaml` | rollout 默认配置，包含 `name`、采样参数、TP、response curriculum 映射等 |
| `verl/trainer/config/reward_manager.yaml` | reward manager 默认配置 |
| `verl/trainer/config/reward_model/reward_model.yaml` | reward model / reward loop 默认配置 |
| `verl/trainer/constants_ppo.py` | Ray runtime env 默认注入环境变量 |
| `verl/trainer/ppo/ray_trainer.py` | 训练主循环；初始化 worker、验证、rollout、reward、advantage、actor/critic update、checkpoint |
| `verl/trainer/ppo/utils.py` | 定义 `Role`，以及是否需要 critic / ref / RM 的判定 |
| `verl/trainer/ppo/reward.py` | reward manager 加载与 reward 计算入口 |
| `verl/protocol.py` | `DataProto` 协议定义，driver 和 worker 之间传输 batch 的核心载体 |
| `verl/utils/dataset/rl_dataset.py` | 默认 RL dataset，把 parquet 样本转成 `raw_prompt` 等字段 |
| `verl/utils/dataset/rlpro_dataset.py` | 你当前脚本实际更相关的数据集；支持 `raw_prompt_response` 和可信响应路径 |
| `verl/experimental/agent_loop/single_turn_agent_loop.py` | 单轮生成逻辑；当前脚本的 curriculum prompt 拼接关键在这里 |
| `verl/experimental/agent_loop/agent_loop.py` | agent loop 总调度；切 batch、调 server、后处理生成结果 |
| `verl/experimental/reward_loop/reward_loop.py` | reward loop worker 管理与 `rm_scores` 回填 |
| `verl/workers/fsdp_workers.py` | FSDP worker 实现；暴露 `generate_sequences`、`compute_log_prob`、`update_actor` 等 RPC |
| `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` | vLLM rollout 核心实现；真正和 vLLM engine 对接 |
| `verl/single_controller/ray/base.py` | Ray worker group / resource pool 封装，负责 dispatch、collect、placement group |

## 6. 配置是怎么落到代码路径上的

### 6.1 Hydra 总配置

`verl/trainer/config/ppo_trainer.yaml` 通过 `defaults` 聚合：

- actor
- data
- reward_manager
- ref
- rollout
- model
- critic
- reward_model
- algorithm

然后 shell 脚本传入的覆盖项会直接改这些节点上的字段。

### 6.2 你这个脚本最关键的覆盖项

#### 训练数据与上下文长度

- `data.train_files=.../XCombined/train.parquet`
- `data.val_files=.../MATH-lighteval/test.parquet`
- `data.train_batch_size=256`
- `data.max_prompt_length=1024`
- `data.max_response_length=39936`

含义：

- 每个训练 step 先从 dataloader 取 256 条样本
- 每条样本 prompt 最长 1024 token
- 最大 response budget 非常大，走的是长上下文训练

#### rollout 设置

- `rollout.name=vllm`
- `rollout.n=8`
- `rollout.temperature=1.4`
- `rollout.top_p=1.0`
- `rollout.max_num_batched_tokens=40960`
- `rollout.tensor_model_parallel_size=2`

含义：

- 每条 prompt rollout 8 次
- 所以 1 个 step 的 rollout 样本数约为 `256 * 8 = 2048`
- rollout 引擎是 vLLM
- rollout 内部 TP=2

#### actor 设置

- `actor.strategy=fsdp`（默认路径）
- `actor.ppo_mini_batch_size=256`
- `actor.ppo_micro_batch_size_per_gpu=2`
- `actor.ulysses_sequence_parallel_size=2`
- `actor.optim.lr=1e-6`
- `actor.use_kl_loss=False`

含义：

- actor 更新走 FSDP worker
- 不启 KL loss
- Ulysses SP=2

#### 算法设置

- `algorithm.adv_estimator=grpo`
- `algorithm.use_kl_in_reward=False`

含义：

- advantage 计算走 GRPO，而不是 GAE
- 不需要 reference policy 参与 reward 修正

## 7. 数据是怎么流的

这部分是理解 verl 的关键。你可以把它看成：

`Parquet -> Dataset -> DataLoader -> DataProto -> Rollout -> Reward -> Advantage -> Actor Update`

下面按 step 展开。

### 7.1 Parquet -> Dataset

入口：`main_ppo.py:create_rl_dataset()`

逻辑：

- 普通训练数据默认用 `RLHFDataset`
- 当 `data.trust_response is not None` 且是训练集时，切到 `RLHFProDataset`

你的脚本设置了：

- `data.trust_response=curriculum`
- `data.response_key=solution`

因此训练集会走：

- `verl/utils/dataset/rlpro_dataset.py`

而验证集仍然是普通 `RLHFDataset`。

### 7.2 RLHFProDataset 输出什么

`RLHFProDataset.__getitem__()` 不是直接输出 tokenized 的 `input_ids`，而是输出更上游的结构化字段，最关键的是：

- `raw_prompt`
- `raw_prompt_response`
- `dummy_tensor`
- `extra_info`
- `index/tools_kwargs/interaction_kwargs`

其中：

- `raw_prompt`：只有用户 prompt 的消息列表
- `raw_prompt_response`：在 prompt 后额外拼了一个“可信 assistant response”的消息列表

这个“可信 response”来自：

- `get_valid_response(example, response_key, is_random=True)`

也就是从样本的 `extra_info[response_key]` 里挑一个可用答案，优先挑 `verification == 1` 的项。

### 7.3 DataLoader -> DataProto

`RayPPOTrainer._create_dataloader()` 使用：

- `StatefulDataLoader`
- `rlpro_dataset.collate_fn` 或 `rl_dataset.collate_fn`

`collate_fn` 的策略很简单：

- tensor 字段堆叠成 tensor
- 非 tensor 字段转成 `numpy.object` 数组

进入训练循环后，`batch_dict` 会被包装为：

- `DataProto.from_single_dict(batch_dict)`

`DataProto` 是整个仓库非常核心的中间协议，定义在：

- `verl/protocol.py`

你可以把它理解为：

- `batch`：tensor 字段
- `non_tensor_batch`：Python/NumPy 对象字段
- `meta_info`：控制信息，例如 `global_steps`、`temperature`、`pad_token_id`

它支持：

- `repeat`
- `union`
- `chunk`
- `concat`
- `reorder`

这些正是 driver 和各类 worker 之间流转 batch 的基础。

### 7.4 训练 step 一开始：repeat n 次

在 `RayPPOTrainer.fit()` 里：

1. dataloader 先给出一批 `batch`
2. 给每个样本打上唯一 `uid`
3. 取出适合生成的 `gen_batch`
4. 按 `rollout.n` 重复

也就是：

- 原始 batch size = 256
- 重复后 rollout 样本数 = 2048

GRPO 后面按 `uid` 分组算 advantage，就是靠这个重复后的同源样本组实现的。

### 7.5 DataProto -> AgentLoop -> vLLM 生成

当前代码路径不是简单直接地调用 vLLM，而是：

1. `RayPPOTrainer.fit()` 调 `self.async_rollout_manager.generate_sequences(...)`
2. `AgentLoopManager.generate_sequences()` 把 batch 切给多个 agent loop worker
3. 每个 worker 执行 `SingleTurnAgentLoop.run()`
4. `SingleTurnAgentLoop` 调 server manager
5. server manager 最终落到 vLLM rollout backend

关键文件：

- `verl/experimental/agent_loop/agent_loop.py`
- `verl/experimental/agent_loop/single_turn_agent_loop.py`
- `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`

生成后会回到统一的 `DataProto`，核心字段通常包括：

- `prompts`
- `responses`
- `input_ids`
- `attention_mask`
- `position_ids`
- `response_mask`
- 可选 `rollout_log_probs`

### 7.6 你这个脚本里的 curriculum 是怎么生效的

这是你当前仓库里最容易看错的一点。

#### 表面上看

`main_ppo.py:create_rl_dataset()` 在 `trust_response=curriculum` 时会构造一个 `curriculum_config`。

#### 但当前主路径里真正生效的不是它

当前 `RLHFProDataset.__init__()` 并不接收 `curriculum_config` 参数；`main_ppo.py` 虽然构造了它，但通过 `build_dataset()` 传参时会把不在签名里的参数丢掉。

也就是说，老版本那种“dataset 内部维护课程状态”的逻辑，在当前主路径里并不是主要生效点。

真正起作用的是 rollout 侧：

- `verl/trainer/config/rollout/rollout.yaml`
  - `response_mode: ${oc.select:data.trust_response,null}`
- `verl/experimental/agent_loop/single_turn_agent_loop.py`
- `verl/experimental/agent_loop/agent_loop.py`
- `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`

也就是：

1. `data.trust_response=curriculum` 会映射到 `rollout.response_mode=curriculum`
2. `SingleTurnAgentLoop.run()` 在训练时会判断是否进入 curriculum 模式
3. 如果进入，就使用 `raw_prompt_response`
4. 也就是把一部分可信答案当成 prompt 前缀喂给模型
5. 同时按 step 动态减少要生成的 token 数

本质上，这更像“response-prefix curriculum”，不是 dataset 采样顺序 curriculum。

### 7.7 reward 是怎么回来的

当前脚本默认：

- `reward_model.enable=False`
- `reward_model.use_reward_loop=True`

于是 reward 路径是：

1. rollout/agent loop 结束后
2. reward loop worker 计算每条样本的 reward
3. `RewardLoopManager.compute_rm_score()` 把分数回填成 `rm_scores`
4. `rm_scores` 被塞回 `DataProto.batch`
5. trainer 再把它赋给 `token_level_scores`

注意 `rm_scores` 的形状是按 response token 对齐的，但通常只在“最后一个有效 response token 位置”填入最终 reward，其它位置为 0。

也就是 reward 更像：

- 稀疏 token-level reward
- 实际上代表 sequence-level outcome reward

对应代码：

- `verl/experimental/reward_loop/reward_loop.py`
- `verl/trainer/ppo/reward.py`

### 7.8 old log prob / ref log prob / value

在你当前脚本里：

- `old_log_probs`：会算
- `ref_log_prob`：不会算
- `values`：默认不会算

原因：

- actor 更新需要 old policy log prob
- 但你禁了 KL 路径，所以 ref policy 不需要
- 你用 GRPO，默认不需要 critic value

对应函数都在 `RayPPOTrainer` 里：

- `_compute_old_log_prob()`
- `_compute_ref_log_prob()`
- `_compute_values()`

底层 RPC 则落到：

- `ActorRolloutRefWorker.compute_log_prob()`
- `ActorRolloutRefWorker.compute_ref_log_prob()`
- `CriticWorker.compute_values()`

### 7.9 GRPO advantage 怎么算

driver 收到 rollout 结果和 reward 后，会在 `ray_trainer.py` 中调用：

- `compute_advantage(..., adv_estimator=GRPO)`

GRPO 分支依赖：

- `token_level_rewards`
- `response_mask`
- `uid`

这里的 `uid` 非常关键，因为同一个原始 prompt 复制出来的 8 个 rollout 样本要按组聚合，GRPO advantage 才有意义。

### 7.10 Actor update

最后 driver 调：

- `_update_actor(batch)`

底层进入：

- `ActorRolloutRefWorker.update_actor()`

这个 worker 会：

1. 必要时把 FSDP 参数/优化器从 CPU offload 拉回 GPU
2. 调 actor 的 `update_policy`
3. 回传 metrics
4. 再按需要 offload 回去

当前脚本最主要的学习更新都在这里完成。

### 7.11 Checkpoint

checkpoint 由 `RayPPOTrainer._save_checkpoint()` 负责，保存内容包括：

- actor checkpoint
- 可选 critic checkpoint
- dataloader state (`data.pt`)
- `latest_checkpointed_iteration.txt`

恢复逻辑在：

- `RayPPOTrainer._load_checkpoint()`

它会：

- 找 `default_local_dir` 下最新 `global_step_*`
- 恢复 actor/critic
- 恢复 dataloader 状态

## 8. 用一张图看完整数据流

```text
Shell script
  -> ray start / ray job submit
  -> python -m verl.trainer.main_ppo
  -> Hydra compose config
  -> TaskRunner.run()
     -> create worker mapping / resource pool
     -> load tokenizer/processor
     -> create train_dataset / val_dataset
     -> create RayPPOTrainer
     -> init_workers()
     -> fit()

fit() per step:
  parquet row
    -> RLHFProDataset.__getitem__
    -> raw_prompt / raw_prompt_response
    -> collate_fn
    -> DataProto(batch, non_tensor_batch, meta_info)
    -> repeat by rollout.n
    -> AgentLoopManager.generate_sequences()
    -> SingleTurnAgentLoop.run()
    -> vLLM generate
    -> DataProto(prompts, responses, input_ids, attention_mask, ...)
    -> RewardLoopManager.compute_rm_score()
    -> token_level_scores / rm_scores
    -> actor compute old_log_probs
    -> driver compute GRPO advantages
    -> actor update
    -> save / validate
```

## 9. Ray 层是怎么把 driver 和 worker 粘起来的

这个仓库最核心的架构思想是：

- 控制流在单进程 driver
- 计算流在多进程 worker

对应核心封装在：

- `verl/single_controller/ray/base.py`

这里面最重要的几个概念：

### 9.1 Role

定义在：`verl/trainer/ppo/utils.py`

比如：

- `ActorRollout`
- `Critic`
- `RefPolicy`
- `RewardModel`
- `ActorRolloutRef`

### 9.2 ResourcePool

定义在：`verl/single_controller/ray/base.py`

用于管理一组 GPU placement group，决定 worker 放在哪些 GPU 上。

### 9.3 WorkerGroup

driver 不直接一张卡一张卡调 worker，而是调 `RayWorkerGroup`。

`RayWorkerGroup` 会负责：

- dispatch 输入数据
- 把 batch 按 DP 切开
- 调各个远程 worker 的 `.remote()`
- 收集输出并 concat 回来

### 9.4 DataProto dispatch

worker 方法上有 `@register(...)` 装饰器，例如：

- `generate_sequences`
- `compute_log_prob`
- `update_actor`

它定义了这类方法应该如何：

- 切分输入
- 分发到各 rank
- 收集结果

这也是为什么 `RayPPOTrainer.fit()` 读起来像单进程代码，但底层实际上是多机多卡分布式执行。

## 10. 你这个脚本里最值得注意的几个实现细节

### 10.1 GRPO 默认不启 critic

虽然 `TaskRunner.run()` 里会先注册 critic worker 类，但真正是否初始化和使用 critic，要看：

- `need_critic(config)`

当：

- `algorithm.adv_estimator=grpo`
- 且 `critic.enable` 没手动设成 `True`

默认就是不启 critic。

所以不要被“代码里出现 CriticWorker”误导，以为这次训练一定用了 value model。

### 10.2 当前 curriculum 主要在 rollout 侧，不在 dataset 内部状态机

当前主路径里：

- `main_ppo.py` 还保留了 `curriculum_config` 构造代码
- `ray_trainer.py` 里也还留着更新 `train_dataset.curriculum_config` 的注释

但真正活跃的 curriculum 逻辑已经转移到：

- `single_turn_agent_loop.py`
- `agent_loop.py`
- `vllm_rollout_spmd.py`

如果你后面想改 curriculum，不应该只盯 dataset。

### 10.3 当前验证集不走 curriculum prompt

`_validate()` 走的是 validation rollout，`SingleTurnAgentLoop.run()` 在 `validate=True` 时直接使用：

- `raw_prompt`

也就是验证不会把 trusted response 前缀塞进 prompt。

这点很重要，因为训练和验证的 prompt 构造方式并不完全相同。

### 10.4 `min_curriculum_epoch` 的实现要自己再确认一遍

当前实现里，curriculum ratio 的公式用了：

- `max(epoch - min_curriculum_epoch, 1)`

这意味着它并不是“在 min epoch 前完全不启用”，而是会从一个接近 `max_response_ratio` 的值开始。

如果你原本预期的是“前 16 step 完全不加 trusted response 前缀”，那当前代码并不是这个行为。

### 10.5 reward loop 即使没 reward model 也会工作

这个仓库里：

- `reward_model.enable=False`
- 不等于没有 reward

因为：

- 仍然可以通过 `reward loop + reward manager` 计算函数式 reward

所以你当前训练不是“无奖励训练”，而是“无独立 reward model，但有 reward loop”。

## 11. 如果你想继续深挖，最值得看的函数

### 11.1 入口与装配

- `verl/trainer/main_ppo.py:main`
- `verl/trainer/main_ppo.py:run_ppo`
- `verl/trainer/main_ppo.py:TaskRunner.run`

### 11.2 主循环

- `verl/trainer/ppo/ray_trainer.py:init_workers`
- `verl/trainer/ppo/ray_trainer.py:fit`
- `verl/trainer/ppo/ray_trainer.py:_validate`
- `verl/trainer/ppo/ray_trainer.py:_compute_old_log_prob`
- `verl/trainer/ppo/ray_trainer.py:_update_actor`

### 11.3 数据与协议

- `verl/utils/dataset/rlpro_dataset.py:__getitem__`
- `verl/protocol.py:DataProto`

### 11.4 rollout

- `verl/experimental/agent_loop/single_turn_agent_loop.py:run`
- `verl/experimental/agent_loop/agent_loop.py:generate_sequences`
- `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py:generate_sequences`

### 11.5 worker RPC

- `verl/workers/fsdp_workers.py:generate_sequences`
- `verl/workers/fsdp_workers.py:compute_log_prob`
- `verl/workers/fsdp_workers.py:update_actor`

## 12. 一句话总结这个仓库的核心设计

verl 的核心不是“某个 PPO 公式实现”，而是这套分层：

- shell / Ray 集群层：负责把作业拉起来
- single-controller driver 层：负责 RL 控制流
- worker group 层：负责把控制流映射到分布式执行
- rollout / actor / reward 层：负责具体计算
- `DataProto` 层：负责把数据在这些组件之间传来传去

对你当前这个脚本，最准确的理解应该是：

`driver 侧组织 GRPO 控制流，FSDP actor 负责训练，vLLM 负责生成，reward loop 负责打分，DataProto 负责在各环节搬运 batch。`
