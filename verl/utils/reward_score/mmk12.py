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
import re

from math_verify import ExprExtractionConfig, LatexExtractionConfig, StringExtractionConfig, parse, verify

def extract_answer_with_tags(text):
    match = re.search(r"(<answer>.*?</answer>)", text)
    if match:
        return match.group(1)
    else:
        try:
            response = text.split("<answer>")[-1]
        except:
            response = text.split("\n")[-1]
    return response

def preprocess(txt: str) -> str:
    # \number{21} → 21
    txt = txt.replace("</answer>", "").replace("<answer>", "").strip()
    txt =  re.sub(r"\\number\s*\{\s*(?P<latex>[^{}]+)\s*\}", r"\1", txt)
    return txt

def format_reward(predict_str: str) -> float:
    pattern = (
        r"^(?=(?:.*<think>){1})(?=(?:.*<\/think>){1})"
        r"(?=(?:.*<answer>){1})(?=(?:.*<\/answer>){1})"
        r"(?!.*<think>.*<think>)"
        r"(?!.*<\/think>.*<\/think>)"
        r"(?!.*<answer>.*<answer>)"
        r"(?!.*<\/answer>.*<\/answer>)"
        r".*<think>(.+?)</think>\s*<answer>.+?</answer>.*$"
    )
    matches = re.search(pattern, predict_str, re.DOTALL)
    return 0.5 if matches else 0.0

def acc_reward(predict_str, ground_truth):
    reward = 0.0
    pred_answer = extract_answer_with_tags(predict_str)
    pred_answer  = preprocess(pred_answer)
    gold_parsed = parse(f"${str(preprocess(ground_truth))}$")
    if len(gold_parsed) != 0:
        answer_parsed = parse(
            pred_answer,
            extraction_config=[StringExtractionConfig(), LatexExtractionConfig(), ExprExtractionConfig()],
        )
        try:
            reward = float(verify(answer_parsed, gold_parsed))
        except Exception:
            pass

        # 逻辑是response里必须出现option字母，仅出现option text不计reward
        if reward == 0.0:
            try:
                choices = ["a", "b", "c", "d"]
                content_match = re.search(r"<answer>(.*?)</answer>", predict_str)
                pred_answer = content_match.group(1).strip() if content_match else content.strip()
                pred_answer  = preprocess(pred_answer)
                for potential_answer in gold_parsed:
                    if str(potential_answer).lower() in choices and str(potential_answer).lower() in pred_answer.lower():
                        choices_other = [choice for choice in choices if choice != str(potential_answer).lower()]
                        if all(choice not in pred_answer.lower() for choice in choices_other):
                            reward = 1.0
            except Exception:
                pass
    else:
        reward = 0.0
        print("Failed to parse gold solution: ", ground_truth)

    return reward



def compute_score(predict_str: str, ground_truth: str, use_boxed: bool = True, format_score: float = 0.1) -> float:
    return acc_reward(predict_str, ground_truth) + format_score * format_reward(predict_str)
