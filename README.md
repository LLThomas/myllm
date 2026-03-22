## Motivation
[tiny-llm](https://github.com/skyzh/tiny-llm) (a course on deploying large language models (LLMs)) is designed for macOS users; it is not very user-friendly for Windows-only users.
This repository has been adapted to work with a specific version of PyTorch, making it easier for Windows-only users to use.

[tiny-llm](https://github.com/skyzh/tiny-llm) (A course on LLM serving) 适合 macos 玩家食用，作为纯 win 党不便于学习。
本仓适配了一个 torch 版本，便于纯 win 党玩耍。

## Usage
1. run inference
* without kvcache
```bash
pdm run main --solution myllm --loader week1 --model qwen2-0.5b --prompt "Give me a short introduction to large language model"
```

* with kvcache
```bash
pdm run main --solution myllm --loader week2 --model qwen2-0.5b --prompt "Give me a short introduction to large language model"
```


2. run one test case
```bash
pdm run pytest test/test_week_1_day_1.py::test_task_1_simple_attention
```

3. run all test cases
```bash
pdm run pytest test/test_week_1_day_1.py -x
```
