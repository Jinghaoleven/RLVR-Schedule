# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import json
import aiohttp
from openai.types.chat import ChatCompletion
from transformers import PreTrainedTokenizer

GRM_PROMPT_TEMPLATE = """
You are given a problem and a proposed solution.

Problem:
{problem}

Solution:
{solution}

Please evaluate how well the solution addresses the problem. 
Give a score from 1 to 10, where:
- 1 means the solution is completely irrelevant or incorrect.
- 5 means the solution is partially correct but incomplete or not well reasoned.
- 10 means the solution is fully correct, well-reasoned, and directly solves the problem.

Only output the score as a single number (integer).
""".strip()


WGT_EXTRACT_PROMPT_TEMPLATE = """
You are given a model solution. Extract only the final answer from the solution.

Solution:
{solution}

Return only the extracted answer text.
Do not explain.
Do not add any extra words.
If the final answer cannot be determined, output [INVALID].
""".strip()


WGT_COMPARE_PROMPT_TEMPLATE = """
You are given a predicted final answer and a ground-truth answer.
Determine whether they are semantically equivalent as final answers to the same question.

Predicted answer:
{pred_answer}

Ground-truth answer:
{ground_truth}

Output only `1` if they are equivalent.
Output only `0` if they are not equivalent.
Do not explain.
Do not output anything else.
""".strip()


async def chat_complete(router_address: str, chat_complete_request: dict):
    url = f"http://{router_address}/v1/chat/completions"
    timeout = aiohttp.ClientTimeout(total=None)
    session = aiohttp.ClientSession(timeout=timeout)
    try:
        async with session.post(url, json=chat_complete_request) as resp:
            output_text = await resp.text()
            try:
                output = json.loads(output_text)
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"Reward model returned non-JSON response. status={resp.status}, body={output_text}"
                ) from e

            if resp.status >= 400:
                raise RuntimeError(
                    f"Reward model request failed. status={resp.status}, response={json.dumps(output, ensure_ascii=False)}"
                )

            if "error" in output:
                raise RuntimeError(
                    f"Reward model returned error payload: {json.dumps(output, ensure_ascii=False)}"
                )

            return ChatCompletion(**output)
    finally:
        await session.close()


async def compute_score_gsm8k(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    reward_router_address: str,
    reward_model_tokenizer: PreTrainedTokenizer,
    model_name: str | None = None,
):
    """Compute the reward score."""

    grm_prompt = GRM_PROMPT_TEMPLATE.format(problem=extra_info["question"], solution=solution_str)
    messages = [{"role": "user", "content": grm_prompt}]
    sampling_params = {"temperature": 0.7, "top_p": 0.8, "max_tokens": 4096}
    if model_name is None:
        model_name = getattr(reward_model_tokenizer, "name_or_path", None)
    if not model_name:
        raise ValueError("model_name is not provided and cannot be inferred from reward_model_tokenizer.name_or_path")
    chat_complete_request = {
        "messages": messages,
        "model": model_name,
        **sampling_params,
    }
    result = await chat_complete(
        router_address=reward_router_address,
        chat_complete_request=chat_complete_request,
    )
    grm_response = result.choices[0].message.content
    try:
        score = int(grm_response.split("\n\n")[-1].strip())
    except Exception:
        score = 0
    return {"score": score, "acc": score == 10, "genrm_response": grm_response}


async def compute_score_gt(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    reward_router_address: str,
    reward_model_tokenizer: PreTrainedTokenizer,
    model_name: str | None = None,
):
    """Extract the final answer with reward model and compare it against ground truth."""
    max_solution_chars = 12000

    def _resolve_model_name() -> str:
        resolved_model_name = model_name
        if resolved_model_name is None:
            resolved_model_name = getattr(reward_model_tokenizer, "name_or_path", None)
        if not resolved_model_name:
            raise ValueError(
                "model_name is not provided and cannot be inferred from reward_model_tokenizer.name_or_path"
            )
        return resolved_model_name

    resolved_model_name = _resolve_model_name()
    truncated_solution = solution_str[-max_solution_chars:] if len(solution_str) > max_solution_chars else solution_str

    extract_prompt = WGT_EXTRACT_PROMPT_TEMPLATE.format(solution=truncated_solution)
    messages = [{"role": "user", "content": extract_prompt}]
    sampling_params = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 128,
    }

    chat_complete_request = {
        "messages": messages,
        "model": resolved_model_name,
        **sampling_params,
    }
    result = await chat_complete(
        router_address=reward_router_address,
        chat_complete_request=chat_complete_request,
    )

    extraction_response = result.choices[0].message.content or ""
    extracted_answer = extraction_response.strip()
    if extracted_answer.startswith("```") and extracted_answer.endswith("```"):
        extracted_answer = "\n".join(extracted_answer.splitlines()[1:-1]).strip()
    extracted_answer = extracted_answer.strip().strip("'").strip('"')
    if "\n" in extracted_answer:
        extracted_answer = extracted_answer.splitlines()[-1].strip()

    normalized_prediction = extracted_answer.strip()
    normalized_ground_truth = str(ground_truth).strip()

    compare_prompt = WGT_COMPARE_PROMPT_TEMPLATE.format(
        pred_answer=normalized_prediction,
        ground_truth=normalized_ground_truth,
    )
    compare_request = {
        "messages": [{"role": "user", "content": compare_prompt}],
        "model": resolved_model_name,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 16,
    }
    compare_result = await chat_complete(
        router_address=reward_router_address,
        chat_complete_request=compare_request,
    )
    compare_response = (compare_result.choices[0].message.content or "").strip()
    try:
        score = float(compare_response.splitlines()[-1].strip())
        if score not in (0.0, 1.0):
            score = 0.0
    except Exception:
        score = 0.0

    return {
        "score": score,
        "acc": bool(score),
        "pred_answer": normalized_prediction,
        "ground_truth": normalized_ground_truth,
        "solution_chars": len(solution_str),
        "truncated_solution_chars": len(truncated_solution),
        "extract_response": extraction_response,
        "compare_response": compare_response,
    }


def compute_score_math_verify(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    **kwargs,
):
    """Compute the reward score."""
    from verl.utils.reward_score.math_verify import compute_score

    return compute_score(
        model_output=solution_str,
        ground_truth=ground_truth,
    )
