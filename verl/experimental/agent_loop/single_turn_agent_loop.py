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

import torch
from PIL import Image

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.diffusion_agent_loop import DiffusionAgentLoopOutput
from verl.utils.chat_template import apply_chat_template
from verl.utils.profiler import simple_timer
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.rollout.replica import TokenOutput

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
            if self.config.actor_rollout_ref.rollout.rollout_strategy == "prefix":
                progress_ratio = self._prefix_curriculum(global_steps)
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
            output: TokenOutput = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                video_data=videos,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        response_mask = [1] * len(output.token_ids)

        output: AgentLoopOutput = AgentLoopOutput(
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
            extra_fields=output.extra_fields,
        )

        # keeping the schema consistent with tool_agent_loop
        output.extra_fields.update({"turn_scores": [], "tool_rewards": []})

        return output


@register("diffusion_single_turn_agent")
class DiffusionSingleTurnAgentLoop(AgentLoopBase):
    """Agent loop for diffusion model serving."""

    # Keys from non_tensor_batch that are pipeline/dataset metadata and must
    # NOT be forwarded to server_manager.generate() (which passes **kwargs
    # down to the vllm-omni server that has a fixed signature).
    _KEYS_EXCLUDED_FROM_GENERATE = frozenset(
        {
            "raw_prompt",
            "raw_negative_prompt",
            "data_source",
            "reward_model",
            "index",
        }
    )

    async def apply_chat_template(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        images: list[Image.Image] | None = None,
        videos: list[tuple[torch.Tensor, dict]] | None = None,
        remove_system_prompt: bool = False,
    ) -> list[int]:
        """Tokenize on the asyncio thread for fast tokenizers when no processor is used.

        Rust-backed fast tokenizers are not reliably safe across ``run_in_executor`` thread
        boundaries with recent transformers (``RuntimeError: Already borrowed``). The diffusion
        path is tokenizer-only for Qwen-Image-style models; keep tokenization on the event-loop
        thread in that case.
        """
        if self.processor is not None:
            return await super().apply_chat_template(
                messages,
                tools=tools,
                images=images,
                videos=videos,
                remove_system_prompt=remove_system_prompt,
            )
        if getattr(self.tokenizer, "is_fast", False):
            tokenized_prompt = apply_chat_template(
                self.tokenizer,
                messages,
                tools=tools,
                add_generation_prompt=True,
                tokenize=True,
                **self.apply_chat_template_kwargs,
            )
            prompt_ids = normalize_token_ids(tokenized_prompt)
            if remove_system_prompt:
                prompt_ids = prompt_ids[len(self.system_prompt) :]
            return prompt_ids
        return await super().apply_chat_template(
            messages,
            tools=tools,
            images=images,
            videos=videos,
            remove_system_prompt=remove_system_prompt,
        )

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> DiffusionAgentLoopOutput:
        raw_prompt = kwargs.pop("raw_prompt")
        raw_negative_prompt = kwargs.pop("raw_negative_prompt", None)
        for key in self._KEYS_EXCLUDED_FROM_GENERATE:
            kwargs.pop(key, None)

        # 1. extract images and videos from messages
        multi_modal_data = await self.process_vision_info(raw_prompt)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        # 2. apply chat template and tokenize
        prompt_ids = await self.apply_chat_template(raw_prompt, images=images, videos=videos)

        if raw_negative_prompt is not None:
            negative_prompt_ids = await self.apply_chat_template(raw_negative_prompt, images=images, videos=videos)
        else:
            negative_prompt_ids = None

        # 3. generate sequences
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                video_data=videos,
                negative_prompt_ids=negative_prompt_ids,
                **kwargs,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1

        output = DiffusionAgentLoopOutput(
            prompt_ids=prompt_ids,
            response_diffusion_output=output.diffusion_output,
            response_logprobs=output.log_probs,
            multi_modal_data=multi_modal_data,
            num_turns=2,
            metrics=metrics,
            extra_fields=output.extra_fields,
        )
        return output
    
    def _prefix_curriculum(self, global_steps: int) -> float:
        # 每次读取 proxy，而不是在 __init__ 时复制到本地
        self.max_model_len = self.config.actor_rollout_ref.rollout.max_model_len or self.config.actor_rollout_ref.rollout.prompt_length + self.config.actor_rollout_ref.rollout.response_length

        def linear_ratio(epoch: int, max_prefix_ratio: float, max_prefix_epoch: int, min_prefix_epoch: int) -> float:
            """
            linear schedule:
            - epoch = 0             → ratio = max_prefix_ratio
            - epoch >= max_prefix_epoch    → ratio = 1.0
            - 0 < epoch < end_ep    → ratio = linear between max_prefix_ratio and 1.0
            """
            return max_prefix_ratio - max_prefix_ratio * (min(max(epoch-min_prefix_epoch,1) / max_prefix_epoch, 1))
            # return max_prefix_ratio - max_prefix_ratio * (min(epoch / max_prefix_epoch, 1))

        progress_ratio = linear_ratio(global_steps, self.config.actor_rollout_ref.rollout.max_prefix_ratio, self.config.actor_rollout_ref.rollout.max_prefix_epoch, self.config.actor_rollout_ref.rollout.min_prefix_epoch)

        return progress_ratio
