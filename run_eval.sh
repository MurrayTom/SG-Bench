# CUDA_VISIBLE_DEVICES=1 python main_test.py \
#     --model_name chatgpt \
#     --openai_model_name gpt-3.5-turbo \
#     --dataset_name ActorAttack \
#     --eval_task original_query \
#     --multi_turn \
#     --attack


CUDA_VISIBLE_DEVICES=1 python main_test.py \
    --model_name qwen2-7b-instruct \
    --model_path /home/myt/Models/Qwen2-7B-Instruct \
    --dataset_name SG-Bench \
    --eval_task original_query \
    --attack

# CUDA_VISIBLE_DEVICES=0 python main_test.py \
#     --model_name qwen1.5-7b-chat \
#     --model_path /home/myt/Models/Qwen1.5-7B-chat \
#     --dataset_name AdvBench520 \
#     --eval_task original_query \
#     --evaluation

# CUDA_VISIBLE_DEVICES=1 python main_test.py \
#     --model_name llama2-13b-chat \
#     --model_path /home/myt/Models/LLAMA2-13B-chat \
#     --dataset_name DivSafe \
#     --eval_task original_query \
#     --evaluation


# CUDA_VISIBLE_DEVICES=5 python main_test.py \
#     --model_name llama3.1-8b-instruct \
#     --model_path /home/myt/Models/LLAMA3.1-8B-Instruct \
#     --dataset_name DivSafe \
#     --eval_task original_query \
#     --attack

# CUDA_VISIBLE_DEVICES=3 python main_test.py \
#     --model_name qwen2-7b-instruct \
#     --model_path /home/myt/Models/Qwen2-7B-Instruct \
#     --dataset_name DivSafe \
#     --eval_task jailbreak_attack \
#     --evaluation

# CUDA_VISIBLE_DEVICES=3 python main_test.py \
#     --model_name qwen2-7b-instruct \
#     --model_path /home/myt/Models/Qwen2-7B-Instruct \
#     --dataset_name DivSafe \
#     --eval_task multiple_choice \
#     --evaluation

# CUDA_VISIBLE_DEVICES=3 python main_test.py \
#     --model_name qwen2-7b-instruct \
#     --model_path /home/myt/Models/Qwen2-7B-Instruct \
#     --dataset_name DivSafe \
#     --eval_task safety_judgement \
#     --evaluation
