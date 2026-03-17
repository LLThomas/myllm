## Motivation
[tiny-llm](https://github.com/skyzh/tiny-llm) (A course on LLM serving) 适合 macos 玩家食用，作为纯 win 党不便于学习。
本仓适配了一个 torch 版本，便于纯 win 党玩耍。

## Usage
1. run inference
pdm run main --solution myllm --loader week1 --model qwen2-0.5b --prompt "Give me a short introduction to large language model"

2. run one case
pdm run pytest test/test_week_1_day_1.py::test_task_1_simple_attention

3. run all test cases
pdm run pytest test/test_week_1_day_1.py -x 