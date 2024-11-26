# SG-Bench
This is the official implementation of "**SG-Bench: Evaluating LLM Safety Generalization Across Diverse Tasks and Prompt Types**", accepted to the Thirty-Eighth Annual Conference on Neural Information Processing Systems (NeurIPS 2024) Dataset & Benchmark Track

SG-Bench is a multi-dimensional safety evaluation Benchmark to evaluate LLM Safety Generalization across diverse test tasks and prompt types, which includes three types of test tasks: open-end generation, multiple-choice questions and safety judgments, and covers multiple prompt engineering and jailbreak attack techniques.

## 📚 Resources
- **[Paper](https://arxiv.org/abs/2410.21965):** Details the evaluation benchmark design and key experimental results.

## Contents
- [Install](#install)
- [Usage](#Usage)

## Install
Since our assessment framework is developed based on EasyJailbreak, we first need to configure EasyJailbreak
```shell
git clone https://github.com/EasyJailbreak/EasyJailbreak.git
cd EasyJailbreak
pip install -e .
```

Next, you need to install some packages. We suggest that the python version >= 3.9
```shell
pip install -r requirement
```

## Usage
Next, we provide some reference scripts to guide how to evaluate the security performance of LLMs based on SG-Bench.

### Open-end Generation
```
CUDA_VISIBLE_DEVICES=1 python main_test.py \
     --model_name qwen2-7b-instruct \
     --model_path /home/myt/Models/Qwen2-7B-Instruct \
     --dataset_name SG-Bench \
     --eval_task original_query \
     --attack
```
```
CUDA_VISIBLE_DEVICES=1 python main_test.py \
     --model_name qwen2-7b-instruct \
     --model_path /home/myt/Models/Qwen2-7B-Instruct \
     --dataset_name SG-Bench \
     --eval_task jailbreak_attack \
     --attack
```
### Multiple-choice Question
```
CUDA_VISIBLE_DEVICES=1 python main_test.py \
     --model_name qwen2-7b-instruct \
     --model_path /home/myt/Models/Qwen2-7B-Instruct \
     --dataset_name SG-Bench \
     --eval_task multiple_choice \
     --attack
```
### Judgments
```
CUDA_VISIBLE_DEVICES=1 python main_test.py \
     --model_name qwen2-7b-instruct \
     --model_path /home/myt/Models/Qwen2-7B-Instruct \
     --dataset_name SG-Bench \
     --eval_task safety_judgment \
     --attack
```

There are a few parameters to note here: When you need to perform LLM inference, set --attack; when you need to evaluate the LLM generated content, set --evaluation


## 🖊️ Citing Info

```bibtex
@article{mou2024sg,
  title={SG-Bench: Evaluating LLM Safety Generalization Across Diverse Tasks and Prompt Types},
  author={Mou, Yutao and Zhang, Shikun and Ye, Wei},
  journal={arXiv preprint arXiv:2410.21965},
  year={2024}
}
```

