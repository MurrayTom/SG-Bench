# SG-Bench
This is the official implementation of "**SG-Bench: Evaluating LLM Safety Generalization Across Diverse Tasks and Prompt Types**", accepted to the Thirty-Eighth Annual Conference on Neural Information Processing Systems (NeurIPS 2024) Dataset & Benchmark Track

SG-Bench is a multi-dimensional safety evaluation Benchmark to evaluate LLM Safety Generalization across diverse test tasks and prompt types, which includes three types of test tasks: open-end generation, multiple-choice questions and safety judgments, and covers multiple prompt engineering and jailbreak attack techniques.


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
