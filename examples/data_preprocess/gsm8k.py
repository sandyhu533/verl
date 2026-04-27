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
"""
Preprocess the GSM8k dataset to parquet format.

功能：把 HuggingFace GSM8K (openai/gsm8k, "main" 子集) 从原始
{question, answer} 两列格式，转换成 veRL 数据层要求的 5 列 schema：
  - data_source     : reward_fn 注册表的路由键
  - prompt          : chat 格式 list[dict]，含 role/content
  - ability         : 指标分组标签（此处固定 "math"）
  - reward_model    : {style, ground_truth}，RLVR 场景的标准答案
  - extra_info      : 诊断信息，不送入模型

输出：train.parquet / test.parquet 写入 ~/data/gsm8k/
被谁使用：run_qwen2-7b.sh 里 data.train_files / data.val_files 直接指向这里的产物。
"""

import argparse
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs


def extract_solution(solution_str):
    """从 GSM8K 原始 answer 字段里抽取最终数值答案。

    GSM8K 的 answer 字段是"完整推理过程 + '#### 最终答案'"的格式，例如：
        "Janet eats 3 eggs... so 16-3-4=9 eggs left.\n#### 9"

    正则 `#### (\-?[0-9\.\,]+)` 匹配 '#### ' 后面的数字（允许负号、小数、千分位逗号），
    然后 split + replace 去掉逗号，返回纯数字字符串如 "9" 或 "1500"。
    这个字符串后续会作为 RLVR 的 ground_truth 给规则式 RM 打分用。
    """
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None                     # 原始数据都带 "####" 标记，没有就是数据异常
    final_solution = solution.group(0)              # 含前缀的完整匹配，例如 "#### 1,500"
    final_solution = final_solution.split("#### ")[1].replace(",", "")  # 去前缀、去千分位逗号 → "1500"
    return final_solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/gsm8k", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "openai/gsm8k"

    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path, "main")
    else:
        dataset = datasets.load_dataset(data_source, "main")

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    instruction_following = 'Let\'s think step by step and output the final answer after "####".'

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        """返回一个 datasets.map(...) 用的闭包。

        闭包参数 `split`（train/test）用于在 extra_info 里打标签，方便后续追踪数据来源。
        内层 process_fn 的签名 `(example, idx)` 配合 map 的 with_indices=True，
        让 idx 作为 extra_info['index']，便于 shuffle 之后反查原始行。
        """

        def process_fn(example, idx):
            # 1. 从原 example 弹出 "question"（弹出而非读取，避免输出里重复保留）
            question_raw = example.pop("question")

            # 2. 拼接 CoT 提示："Let's think step by step and output the final answer after '####'."
            #    模型推理时会看到 question + instruction_following 作为 prompt
            question = question_raw + " " + instruction_following

            # 3. 弹出 answer（包含完整推理过程 + "#### <数字>" 结尾）
            answer_raw = example.pop("answer")
            # 4. 只抽最终数字作为 ground_truth，供规则式 RM 比对
            solution = extract_solution(answer_raw)

            # 5. 构造 veRL 标准 5 列 schema
            data = {
                "data_source": data_source,          # 固定 "openai/gsm8k"，reward_fn 用此字符串路由
                "prompt": [                          # chat 格式（list[dict]），不是 str
                    {
                        "role": "user",              # GSM8K 单轮：只有 user
                        "content": question,
                    }
                ],
                "ability": "math",                   # 分组标签，wandb 指标按 ability 聚合
                "reward_model": {
                    "style": "rule",                 # style=rule → 调用规则函数打分（而不是神经网络 RM）
                    "ground_truth": solution,        # "9" / "1500" 等纯数字字符串
                },
                "extra_info": {                      # 诊断字段，永不送模型
                    "split": split,                  # "train" / "test"
                    "index": idx,                    # 原始行号，方便调试反查
                    "answer": answer_raw,            # 保留完整推理过程，用于错误分析
                    "question": question_raw,        # 保留未拼 instruction 的原题干
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_save_dir, dst=hdfs_dir)
