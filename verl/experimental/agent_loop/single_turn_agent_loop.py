# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("single_turn_agent")
class SingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length

        tool_config_path = self.config.data.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]

    async def run(self, sampling_params: dict[str, Any], global_steps: int, validate: bool, **kwargs) -> AgentLoopOutput:

        with_response = False
        if validate:
            messages = list(kwargs["raw_prompt"])
        else:
            if self.config.actor_rollout_ref.rollout.response_mode == "curriculum":
                progress_ratio = self._curriculum_params(global_steps)
                with_response = progress_ratio > 0
            if with_response:
                messages = list(kwargs["raw_prompt_response"])
                messages_user = list(kwargs["raw_prompt"])
            else:
                messages = list(kwargs["raw_prompt"])

        # 1. extract images and videos from messages
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        # 2. apply chat template and tokenize
        prompt_ids = await self.apply_chat_template(
            messages,
            tools=self.tool_schemas,
            images=images,
            videos=videos,
            add_generation_prompt=not with_response
        )

        prompt_mask = None
        if with_response:
            prompt_ids_user = await self.apply_chat_template(
                messages_user,
                tools=self.tool_schemas,
                images=images,
                videos=videos,
                add_generation_prompt=True
            )
            prompt_length = len(prompt_ids_user)
            response_length = len(prompt_ids) - len(prompt_ids_user)
            sample_length = prompt_length + round(response_length * progress_ratio)
            prompt_ids = prompt_ids[:sample_length]
            prompt_mask = [0] * prompt_length + [1] * (len(prompt_ids) - prompt_length)

        # 3. generate sequences
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                video_data=videos,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        response_mask = [1] * len(output.token_ids)

        # output = AgentLoopOutput(
        #     prompt_ids=prompt_ids,
        #     prompt_mask=prompt_mask,
        #     response_ids=output.token_ids[: self.response_length],
        #     response_mask=response_mask[: self.response_length],
        #     response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
        #     routed_experts=(
        #         output.routed_experts[: len(prompt_ids) + self.response_length]
        #         if output.routed_experts is not None
        #         else None
        #     ),
        #     multi_modal_data=multi_modal_data,
        #     num_turns=2,
        #     metrics=metrics,
        # )
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            response_ids=output.token_ids,
            response_mask=response_mask,
            response_logprobs=output.log_probs if output.log_probs else None,
            routed_experts=(
                output.routed_experts
                if output.routed_experts is not None
                else None
            ),
            multi_modal_data=multi_modal_data,
            num_turns=2,
            metrics=metrics,
        )
        return output
    
    def _curriculum_params(self, global_steps: int) -> float:
        # 每次读取 proxy，而不是在 __init__ 时复制到本地
        self.max_model_len = self.config.actor_rollout_ref.rollout.max_model_len or self.config.actor_rollout_ref.rollout.prompt_length + self.config.actor_rollout_ref.rollout.response_length

        def linear_ratio(epoch: int, max_response_ratio: float, max_curriculum_epoch: int, min_curriculum_epoch: int) -> float:
            """
            linear schedule:
            - epoch = 0             → ratio = max_response_ratio
            - epoch >= max_curriculum_epoch    → ratio = 1.0
            - 0 < epoch < end_ep    → ratio = linear between max_response_ratio and 1.0
            """
            return max_response_ratio - max_response_ratio * (min(max(epoch-min_curriculum_epoch,1) / max_curriculum_epoch, 1))
            # return max_response_ratio - max_response_ratio * (min(epoch / max_curriculum_epoch, 1))

        progress_ratio = linear_ratio(global_steps, self.config.actor_rollout_ref.rollout.max_response_ratio, self.config.actor_rollout_ref.rollout.max_curriculum_epoch, self.config.actor_rollout_ref.rollout.min_curriculum_epoch)

        return progress_ratio
