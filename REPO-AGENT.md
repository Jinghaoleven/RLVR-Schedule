# verl 里的 agent / tool use / 多轮实现

## 这块能力能做什么

这个仓库的 agent 相关能力，不是独立框架，而是叠在原本 RL 训练链路上的一层“轨迹生成系统”。打开 `agent loop + multi_turn + tool/interactions` 后，主要能做这些事：

- 单轮 RL：prompt -> response -> reward -> PPO/GRPO update
- 多轮 RL：模型生成后继续接收 observation，再继续生成
- Tool-use RL：模型生成 function call，框架执行工具，再把结果回灌给模型
- Environment / interaction RL：环境在每轮给 observation、reward、termination
- 多模态 agent：工具或环境可以返回图片并继续进入后续轮次

一句话：`verl` 把“模型生成 / 工具调用 / 环境交互 / reward”拼成完整轨迹，再用 PPO/GRPO 训练 actor。

## 建议先按 4 层理解

1. 训练层：负责采样、reward/advantage、actor update
2. rollout server 层：把 actor 权重暴露成异步推理服务
3. agent loop 层：负责多轮状态机
4. tool / interaction 层：负责外部动作和外部反馈

对应核心文件：

- `verl/trainer/main_ppo.py`
  - 训练入口
- `verl/trainer/ppo/ray_trainer.py`
  - 主训练循环
- `verl/experimental/agent_loop/agent_loop.py`
  - agent loop 抽象、worker、manager、postprocess、reward 接入点
- `verl/experimental/agent_loop/tool_agent_loop.py`
  - 多轮 tool-use agent 主实现
- `verl/experimental/agent_loop/single_turn_agent_loop.py`
  - 默认单轮 loop
- `verl/experimental/agent_loop/tool_parser.py`
  - tool call 解析
- `verl/tools/base_tool.py`
  - 工具接口
- `verl/tools/utils/tool_registry.py`
  - 工具注册与配置加载
- `verl/interactions/base.py`
  - 环境交互接口
- `verl/interactions/utils/interaction_registry.py`
  - interaction 注册与配置加载
- `verl/workers/config/rollout.py`
  - multi-turn / agent 配置定义
- `verl/trainer/config/rollout/rollout.yaml`
  - rollout 用户侧主配置入口
- `verl/workers/rollout/replica.py`
  - rollout replica 抽象
- `verl/checkpoint_engine/base.py`
  - trainer 和 rollout server 的权重同步桥

## actor model 和 agent 是怎么连起来的

这里最关键的一点：agent loop 每一轮不是直接调 trainer 里的 actor forward，而是调异步 rollout server。

主链路是：

1. trainer 更新 actor 权重
2. `checkpoint_manager.update_weights()` 同步到 rollout replicas
3. `AgentLoopWorker` 通过 `AsyncLLMServerManager` 访问 rollout server
4. `ToolAgentLoop` 每一轮调用 `server_manager.generate(...)`
5. rollout server 返回 token，agent loop 再决定是否继续调工具或进环境
6. 完整轨迹回到 trainer，继续做 logprob / reward / advantage / PPO/GRPO update

所以可以把它理解成：

- `agent loop` 负责生成轨迹
- `trainer` 负责消费轨迹并训练 actor

## 一条样本的实际数据流

1. 数据集给出样本，通常包含 `raw_prompt` / `extra_info` / 可选 `agent_name`
2. `AgentLoopManager.generate_sequences()` 把 batch 分给多个 `AgentLoopWorker`
3. 每个 sample 进入 `_run_agent_loop(...)`
4. 依据 `agent_name` 选择具体 loop，没有则用 `default_agent_loop`
5. loop 把 `raw_prompt` 通过 chat template 变成 `prompt_ids`
6. 调 rollout server 生成 assistant token
7. 若生成中出现 tool call，则执行工具，把结果追加到消息历史
8. 若配置 interaction，则向环境拿下一轮 observation / reward
9. 循环直到达到终止条件
10. `_agent_loop_postprocess()` 把轨迹整理成训练张量
11. 若 loop 自己没给最终 reward，则异步 reward worker 再打分
12. trainer 用这些张量做 PPO/GRPO 更新

训练时关键张量通常有：

- `prompts`
- `responses`
- `input_ids`
- `attention_mask`
- `position_ids`
- `response_mask`
- `rm_scores`

## `response_mask` 是这块最重要的字段

在多轮 agent 轨迹里，`responses` 不是纯模型输出，而是“模型 token + observation token”的混合序列。

在这个仓库里：

- 模型生成 token：`response_mask = 1`
- tool 返回 token：`response_mask = 0`
- interaction / 环境返回 token：`response_mask = 0`
- padding：`0`

这决定了哪些 token 真正算 actor 的 action，哪些只是外部 observation。

如果你调 agent 训练效果，最后一定要回到 `responses + response_mask` 上看。

## agent loop 这一层怎么组织

`AgentLoopBase` 是统一抽象，负责：

- 持有 tokenizer / processor
- `apply_chat_template`
- 处理 vision 输入
- 定义统一 `run()` 接口

具体 loop 通过 registry 管理：

- 内置：`single_turn_agent`、`tool_agent`
- 扩展：`rollout.agent.agent_loop_config_path`

运行时按样本里的 `agent_name` 选择 loop；没有则走：

- `actor_rollout_ref.rollout.agent.default_agent_loop`

## `tool_agent` 的核心状态机

`verl/experimental/agent_loop/tool_agent_loop.py` 是主实现。它内部是一个状态机：

- `PENDING`
- `GENERATING`
- `PROCESSING_TOOLS`
- `INTERACTING`
- `TERMINATED`

大致逻辑：

1. `PENDING`
   - 把 messages 转成初始 `prompt_ids`
2. `GENERATING`
   - 调 rollout server 继续生成
   - 更新 `prompt_ids` / `response_mask`
   - 尝试解析 tool call
3. `PROCESSING_TOOLS`
   - 并发执行工具
   - 生成 `role="tool"` 消息并回灌到上下文
   - 这些 token 进入轨迹，但 `response_mask = 0`
4. `INTERACTING`
   - 调 interaction 生成下一轮 observation
   - observation 也进入轨迹，但 `response_mask = 0`
5. `TERMINATED`
   - 到达 token 上限、轮数上限，或 interaction 要求终止

终止主要受这些配置控制：

- `response_length`
- `max_assistant_turns`
- `max_user_turns`

## 多轮历史是如何维护的

`ToolAgentLoop` 内部有个 `AgentData`，保存整条运行时状态，重点包括：

- `messages`
- `prompt_ids`
- `response_ids`
- `response_mask`
- `response_logprobs`
- `turn_scores`
- `tool_rewards`
- `image_data` / `video_data`
- `tool_calls`
- `extra_fields`

可以把它理解成两套视图：

- `messages`：语义上的对话历史，供下一轮 chat template 使用
- `prompt_ids/response_mask`：训练视角下的 token 历史，供 PPO/GRPO 使用

## agent 环境怎么配置

这块最常见的必要配置是：

- `data.return_raw_chat=True`
- `actor_rollout_ref.rollout.mode=async`
- `actor_rollout_ref.rollout.multi_turn.enable=True`

常见相关配置在：

- `verl/workers/config/rollout.py`
- `verl/trainer/config/rollout/rollout.yaml`

`multi_turn` 里最关键的项：

- `enable`
- `max_assistant_turns`
- `max_user_turns`
- `tool_config_path`
- `interaction_config_path`
- `max_parallel_calls`
- `max_tool_response_length`
- `tool_response_truncate_side`
- `format`

`agent` 相关配置：

- `rollout.agent.num_workers`
- `rollout.agent.default_agent_loop`
- `rollout.agent.agent_loop_config_path`
- `rollout.agent.custom_async_server`
- `rollout.agent.agent_loop_manager_class`

## 数据集侧至少要准备什么

agent 相关问题很多不在 trainer，而在数据字段。

通常至少要有：

- `raw_prompt`
  - 多轮消息列表，agent loop 真正读取它
- `extra_info`
  - 样本级附加参数

常见可选字段：

- `agent_name`
  - 指定具体 loop，例如 `tool_agent`
- `extra_info.tools_kwargs`
  - 每个工具的样本级参数
- `extra_info.interaction_kwargs`
  - interaction 的样本级参数，通常必须带 `name`

如果没给 `agent_name`，大概率不会走你想要的 tool agent。

## tool use 是怎么落地的

### 1. 先由模型生成 tool call

tool 调用不是框架替模型决定的，而是模型先按约定格式输出 function call。

解析逻辑在：

- `verl/experimental/agent_loop/tool_parser.py`

当前主支持格式：

- `hermes`
- `gpt-oss`

### 2. 再由框架执行工具

`ToolAgentLoop._call_tool(...)` 的链路是：

1. 按名字查找工具
2. `tool.create(...)`
3. `tool.execute(...)`
4. `tool.release(...)`

工具执行时会拿到 `agent_data`，所以工具能看到整条上下文，不只是函数参数。

### 3. 工具如何回到轨迹里

工具结果会被转成 `role="tool"` 的消息，再 tokenization 回灌给模型。

这些 token：

- 会进入 `prompt_ids`
- 但不会当作 actor action 训练，即 `response_mask = 0`

### 4. 当前支持哪些工具

仓库里能直接看到的工具包括：

- `Gsm8kTool`
- `Geo3kTool`
- `SearchTool`
- `SandboxFusionTool`
- `ImageZoomInTool`
- `MCPSearchTool`

工具分两类：

- native tool：本地 Python 类实现
- MCP tool：通过 MCP client 动态接入

工具是否真的可调用，关键看两件事：

- 有没有写进 `tool_config_path` 对应的 YAML
- 有没有暴露给模型的 tool schema

## interaction / 环境是怎么落地的

interaction 可以理解成“环境 step”。它不要求模型先生成 tool call，而是在一轮模型生成后，由 agent loop 主动调用。

接口在：

- `verl/interactions/base.py`

关键方法：

- `start_interaction(...)`
- `generate_response(...)`
- `calculate_score()`
- `finalize_interaction()`

其中最关键的是 `generate_response(...)`，它返回：

- 是否终止
- 下一轮文本 observation
- 当前 turn reward
- 额外信息

仓库里现成例子：

- `Gsm8kInteraction`
- `WeatherInteraction`
- recipe 里还有 `CollabLLMInteraction`

interaction 配置同样来自 YAML，样本再通过 `extra_info.interaction_kwargs.name` 选择具体环境。

## tool 和 interaction 的分工

建议这么理解：

- tool：模型主动发起的外部调用
- interaction：环境反馈或环境推进

在实现上：

- tool 触发条件是模型先生成了 function call
- interaction 触发条件是 agent loop 主动进入 `INTERACTING`
- tool response 通常用 `role="tool"`
- interaction response 在当前实现里通常作为新的 `user` message 追加

## reward 在 agent 场景里的位置

这块有两层 reward：

### 1. turn-level reward

来自 interaction 或工具：

- interaction reward -> `turn_scores`
- tool reward -> `tool_rewards`

### 2. trajectory-level reward

训练更关心最终整条轨迹 reward。

在 `AgentLoopWorker._compute_score(...)` 里：

- 如果 loop 自己已经给了 `reward_score`，直接用
- 否则如果启用了 reward loop worker，就把单条轨迹发出去计算

最终 reward 会被写成 `rm_scores`，并落在“最后一个有效 response token”上。

## 现成示例从哪里看最快

如果你要快速理解这块，优先看这些文件：

- `docs/start/agentic_rl.rst`
- `docs/advance/agent_loop.rst`
- `docs/sglang_multiturn/multiturn.rst`
- `docs/sglang_multiturn/interaction_system.rst`
- `examples/sglang_multiturn/README.md`
- `examples/sglang_multiturn/config/gsm8k_multiturn_grpo.yaml`
- `examples/sglang_multiturn/config/gsm8k_multiturn_grpo_w_interaction.yaml`
- `examples/sglang_multiturn/config/tool_config/*.yaml`
- `examples/sglang_multiturn/config/interaction_config/*.yaml`
- `examples/data_preprocess/gsm8k_tool_agent_loop.py`
- `examples/data_preprocess/gsm8k_multiturn_w_tool.py`

## 你如果要扩展，主要有 3 个落点

- 新 agent loop
  - 继承 `AgentLoopBase`，实现新的多轮状态机
- 新 tool
  - 继承 `BaseTool` 或 MCP 相关基类，提供 schema 和执行逻辑
- 新 interaction
  - 继承 `BaseInteraction`，定义环境反馈和 turn-level reward

## 对使用最有帮助的几个判断

- 先分清你需要的是 tool 还是 interaction
- 先查数据字段，再查 trainer
- 调 agent 训练问题时，最后一定看 `responses` 和 `response_mask`
- 如果工具结果看起来不完整，先检查 `max_tool_response_length`
- 图片链路已支持，video 在 tool response 主路径里还不完整
- 多轮问题建议开 tracing，不然轨迹很难复盘

## 一份最小心智模型

- `trainer` 训练 actor
- `rollout server` 提供异步生成服务
- `agent loop` 把生成、工具、环境串成轨迹
- `tool` 和 `interaction` 连接外部世界
- `response_mask` 决定哪些 token 真正被训练

抓住这 5 点，再回头读具体文件，会顺很多。
