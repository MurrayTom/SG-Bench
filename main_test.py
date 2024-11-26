import argparse

from easyjailbreak.attacker.Multilingual_Deng_2023 import Multilingual
from easyjailbreak.attacker.Jailbroken_wei_2023 import Jailbroken
from easyjailbreak.attacker.AutoDAN_Liu_2023 import AutoDAN
from easyjailbreak.attacker.MCQ import MultipleChioceQuestion, MultipleChioceQuestion_evaluation
#from easyjailbreak.attacker.MCQ_v2 import MultipleChioceQuestion, MultipleChioceQuestion_evaluation
from easyjailbreak.attacker.SelfEval import selfevalcheck, selfevalcheck_evaluation, claude3_check_dataset
from easyjailbreak.datasets import JailbreakDataset
from easyjailbreak.models.openai_model import OpenaiModel
from easyjailbreak.models.huggingface_model import from_pretrained

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default="llama2-7b-chat", type=str, required=True)
parser.add_argument('--model_path', default="/home/myt/Models/LLAMA2-7B-chat", type=str)
parser.add_argument('--tokenizer_path', default=None, type=str)
parser.add_argument('--dataset_name', default="DivSafe", type=str, required=True)
parser.add_argument('--eval_task', default="original_query", type=str, choices=["original_query", "jailbreak_attack", "multiple_choice", "safety_judgement"], required=True)
parser.add_argument('--prompt_mode', default="", type=str, choices=["", "ToP", "RoP", "RoP_fewshot_general", "COT", "fewshot"])

parser.add_argument('--openai_model_name', default="", type=str)
parser.add_argument('--api_key', default="sk-viv3lQLoCNWGY6gj4cE78bF1D077491f9221A0E11d2f2986", type=str)
parser.add_argument('--eval_by_gpt4', action="store_true", default=False)
parser.add_argument('--multi_turn', action="store_true", default=False)
parser.add_argument('--judge_model_path', default="/home/myt/Models/Llama-Guard-3-8B", type=str)
parser.add_argument("--attack", action="store_true", default=False)
parser.add_argument("--evaluation", action="store_true", default=False)

args = parser.parse_args()

supprted_dataset_list = {
    "AdvBench520": ["/home/myt/Datasets/Eval_benchmark/AdvBench", "/home/myt/Datasets/Eval_benchmark/AdvBench-MCQ/AdvBench-mcq-test.json", "/home/myt/Datasets/Eval_benchmark/AdvBench-SelfEval/AdvBench-SelfEval-test.json", 520],
    "SG-Bench": ["/home/myt/DivSafe-master/datasets/SG-Bench/original_query", "/home/myt/DivSafe-master/datasets/SG-Bench/original_query", "/home/myt/DivSafe-master/datasets/SG-Bench/mcq_test/mcq_test.json", "/home/myt/DivSafe-master/datasets/SG-Bench/judge_test/judge_test.json", 1442],
    "SaladBench": ["/home/myt/DivSafe-master/datasets/SaladBench/base_set", "/home/myt/DivSafe-master/datasets/SaladBench/attack_enhanced_set", "", "", ""],
    "ALERT": ["/home/myt/DivSafe-master/datasets/ALERT/alert_raw", "/home/myt/DivSafe-master/datasets/ALERT/alert_adv", "", "", ""],
    "ActorAttack": ["/home/myt/DivSafe-master/datasets/ActorAttack", "", "", "", ""]
}

supported_model_list = ["chatgpt", "gpt4", "claude3", "o1-mini", "gpt-4o", "mistral-7b-instruct", "llama3-8b-instruct", "llama3.1-8b-instruct", "llama3-8b-instruct-safeprompt", "llama2-13b-chat", "llama2-7b-chat", "llama2-7b-chat_gcg", "llama2-7b-chat_fair", "qwen2-7b-instruct", "qwen1.5-14b-chat", "qwen1.5-7b-chat", "qwen1.5-7b-chat_gcg", "qwen1.5-7b-chat_fair", "chatglm3-6b", "internlm2-7b-chat", "qwen-7b-chat"]
#my_model_list = ["llama3-8b-instruct-SRT", "llama3-8b-instruct-SRT-2", "llama3-8b-instruct-SRT-3", "llama3-8b-instruct-SFT", "llama3-8b-instruct-SFTdivsafe", "qwen2-7b-instruct-SRT", "qwen2-7b-instruct-SRT-2", "qwen2-7b-instruct-SFT", "qwen2-7b-instruct-sftdivsafe"]
my_model_list = ["llama3-8b-multitask_safety_v1", "llama3-8b-COT_safety_v1", "llama3-8b-COT_safety_v2"]

# model_name = "llama2-7b-chat"
# eval_by_gpt4 = True
# dataset_name = "DivSafe"
# eval_set = "Non-adv" # Non-adv, JB, MCQ, TFQ
# MODE = "attack" # attack, evaluation
# prompt_mode = ""
# scene_mode = "single"

if args.attack:
    if "mistral" in args.model_name or "llama2" in args.model_name:
        from easyjailbreak.models.huggingface_model import from_pretrained
    elif "llama3" in args.model_name:
        from easyjailbreak.models.llama3_models import from_pretrained
    elif "qwen-" in args.model_name:
        from easyjailbreak.models.modelscope_model import from_pretrained
    elif "chatglm3" in args.model_name or "internlm2" in args.model_name:
        from easyjailbreak.models.glm_model import from_pretrained
    elif "qwen1.5" in args.model_name or "qwen2" in args.model_name:
        from easyjailbreak.models.qwen2_model import from_pretrained

if args.eval_by_gpt4:
    eval_model = OpenaiModel(model_name='gpt-4-turbo',
                         api_keys=args.api_key)
else:
    eval_model = from_pretrained(model_name_or_path=args.judge_model_path,
                                   model_name='llama-2')

if args.eval_task == "original_query":
    dataset = JailbreakDataset(supprted_dataset_list[args.dataset_name][0])
elif args.eval_task == "jailbreak_attack":
    dataset = JailbreakDataset(supprted_dataset_list[args.dataset_name][1])
print(len(dataset))

if args.attack:
    print("-----------Attack Mode-----------")
    print("target model:", args.model_name)
    
    if args.model_name == "chatgpt":
        target_model = OpenaiModel(model_name=args.openai_model_name,
                            api_keys=args.api_key)
    elif args.model_name == "gpt4":
        target_model = OpenaiModel(model_name=args.openai_model_name,
                            api_keys=args.api_key)
    elif args.model_name == "claude3":
        target_model = OpenaiModel(model_name=args.openai_model_name,
                            api_keys=args.api_key)
    elif args.model_name == "gpt-4o":
        target_model = OpenaiModel(model_name=args.openai_model_name,
                            api_keys=args.api_key)
    elif args.model_name == "o1-mini":
        target_model = OpenaiModel(model_name=args.openai_model_name,
                            api_keys=args.api_key)
    elif args.model_name in supported_model_list:
        target_model = from_pretrained(model_name_or_path=args.model_path,
                                    model_name='llama-2')
    elif args.model_name in my_model_list:
        target_model = from_pretrained(model_name_or_path=args.model_path,
                                    model_name='llama-2')
    else:
        print("model does not implement yet.....")
        exit()
elif args.evaluation:
    print("-----------Evaluation Mode-----------")
    target_model = eval_model
else:
    print("The mode is not supported ....")
    exit()

if args.eval_task == "original_query":
    attacker = Multilingual(attack_model=target_model,
                    target_model=target_model,
                    eval_model=eval_model,
                    eval_by_gpt4=args.eval_by_gpt4,
                    jailbreak_datasets=dataset,
                    multi_turn = args.multi_turn)
    
    save_path_template = "./outputs_results_benchmark/llm_response/{dataset_name}/{prompt_mode}/{dataset_name}-{model_name}.json"
    eval_path_template = "./outputs_results_benchmark/eval_results/{dataset_name}/{prompt_mode}/{dataset_name}-{model_name}_eval.json"
    
    if args.attack:
        save_path = save_path_template.format(dataset_name=args.dataset_name, prompt_mode=args.prompt_mode, model_name=args.model_name)
        print("save_path:", save_path)
        attacker.attack(save_path, args.prompt_mode)
    elif args.evaluation:
        save_path = save_path_template.format(dataset_name=args.dataset_name, prompt_mode=args.prompt_mode, model_name=args.model_name)
        eval_path = eval_path_template.format(dataset_name=args.dataset_name, prompt_mode=args.prompt_mode, model_name=args.model_name)
        attacker.evaluation(input_path = save_path, eval_output_path = eval_path)

elif args.eval_task == "jailbreak_attack":
# Then instantiate the recipe.
    attacker = Jailbroken(attack_model=target_model,
                    target_model=target_model,
                    eval_model=eval_model,
                    eval_by_gpt4=args.eval_by_gpt4,
                    jailbreak_datasets=dataset)

    # attacker = AutoDAN(attack_model=target_model,
    #                 target_model=target_model,
    #                 eval_model=eval_model,
    #                 eval_by_gpt4=args.eval_by_gpt4,
    #                 jailbreak_datasets=dataset)

    save_path_template = "./outputs_results_benchmark/llm_response_jailbreak/{dataset_name}/{prompt_mode}/{dataset_name}_jailbreak-{model_name}.json"
    eval_path_template = "./outputs_results_benchmark/eval_results_jailbreak/{dataset_name}/{prompt_mode}/{dataset_name}_jailbreak-{model_name}_eval.json"
        
    if args.attack:
        save_path = save_path_template.format(dataset_name=args.dataset_name, prompt_mode=args.prompt_mode, model_name=args.model_name)
        attacker.attack(save_path, args.prompt_mode)
    elif args.evaluation:
        save_path = save_path_template.format(dataset_name=args.dataset_name, prompt_mode=args.prompt_mode, model_name=args.model_name)
        eval_path = eval_path_template.format(dataset_name=args.dataset_name, prompt_mode=args.prompt_mode, model_name=args.model_name)
        print(save_path)
        print(eval_path)
        ASR = attacker.evaluation(input_path = save_path, eval_output_path = eval_path)
        print(args.dataset_name+"/"+args.model_name+"/"+args.eval_task)
        print("jailbreak attack success rate:", str(ASR))

elif args.eval_task == "multiple_choice":
    save_path_template = "./outputs_results_benchmark/llm_response_mcq/{dataset_name}/{prompt_mode}/{dataset_name}_MCQ-{model_name}.json"
    eval_path_template_1 = "./outputs_results_benchmark/eval_results_mcq/{dataset_name}/{prompt_mode}/{dataset_name}_MCQ-{model_name}_eval.json"
    eval_path_template_2 = "./outputs_results_benchmark/eval_results_mcq/{dataset_name}/{prompt_mode}/{dataset_name}_MCQ-{model_name}_eval-2.json"
    if args.attack:
        save_path = save_path_template.format(dataset_name=args.dataset_name, prompt_mode=args.prompt_mode, model_name=args.model_name)
        MultipleChioceQuestion(target_model=target_model, test_file=supprted_dataset_list[args.dataset_name][2], save_path=save_path, prompt_mode=args.prompt_mode)
    elif args.evaluation:
        save_path = save_path_template.format(dataset_name=args.dataset_name, prompt_mode=args.prompt_mode, model_name=args.model_name)
        eval_path_1 = eval_path_template_1.format(dataset_name=args.dataset_name, prompt_mode=args.prompt_mode, model_name=args.model_name)
        eval_path_2 = eval_path_template_2.format(dataset_name=args.dataset_name, prompt_mode=args.prompt_mode, model_name=args.model_name)
        false_rate_non_format_1, false_rate_format_1, false_format_nums_1, error_format_nums_1 = MultipleChioceQuestion_evaluation(eval_model=eval_model, results_file=save_path, save_path=eval_path_1)
        false_rate_non_format, false_rate_format, false_format_nums, error_format_nums = MultipleChioceQuestion_evaluation(eval_model=eval_model, results_file=save_path, save_path=eval_path_2, format_mode=False)
        # print("false_rate_non_format / false_rate_format / false_format_nums / error_format_nums")
        # print(false_rate_non_format_1, false_rate_format_1, false_format_nums_1, error_format_nums_1)
        # print("--------------------------------------")
        #print("false_rate_non_format / false_rate_format / false_format_nums / error_format_nums")
        print("multiple-choice question test -- false rate:", false_rate_non_format)


elif args.eval_task == "safety_judgement":
    save_path_template = "./outputs_results_benchmark/llm_response_selfeval/{dataset_name}/{prompt_mode}/{dataset_name}_SelfEval-{model_name}.json"
    eval_path_template_1 = "./outputs_results_benchmark/eval_results_selfeval/{dataset_name}/{prompt_mode}/{dataset_name}_SelfEval-{model_name}_eval.json"
    eval_path_template_2 = "./outputs_results_benchmark/eval_results_selfeval/{dataset_name}/{prompt_mode}/{dataset_name}_SelfEval-{model_name}_eval-2.json"
    if args.attack:
        save_path = save_path_template.format(dataset_name=args.dataset_name, prompt_mode=args.prompt_mode, model_name=args.model_name)
        selfevalcheck(target_model=target_model, test_file=supprted_dataset_list[args.dataset_name][3], save_path=save_path, prompt_mode=args.prompt_mode)
    elif args.evaluation:
        save_path = save_path_template.format(dataset_name=args.dataset_name, prompt_mode=args.prompt_mode, model_name=args.model_name)
        eval_path_1 = eval_path_template_1.format(dataset_name=args.dataset_name, prompt_mode=args.prompt_mode, model_name=args.model_name)
        eval_path_2 = eval_path_template_2.format(dataset_name=args.dataset_name, prompt_mode=args.prompt_mode, model_name=args.model_name)
        false_rate_non_format_1, false_rate_format_1, false_format_nums_1, error_format_nums_1 = selfevalcheck_evaluation(eval_model=eval_model, results_file=save_path, save_path=eval_path_1, Ts=supprted_dataset_list[args.dataset_name][4])
        false_rate_non_format, false_rate_format, false_format_nums, error_format_nums = selfevalcheck_evaluation(eval_model=eval_model, results_file=save_path, save_path=eval_path_2, Ts=supprted_dataset_list[args.dataset_name][4], format_mode=False)
        # print("false_rate_non_format / false_rate_format / false_format_nums / error_format_nums")
        # print(false_rate_non_format_1, false_rate_format_1, false_format_nums_1, error_format_nums_1)
        # print("--------------------------------------")
        #print("false_rate_non_format / false_rate_format / false_format_nums / error_format_nums")
        print("safety judgment test -- false rate:", false_rate_non_format)
    # elif MODE == "check":
    #     save_path = save_path_template.format(dataset_name=dataset_name, model_name=model_name)
    #     eval_path_1 = eval_path_template_1.format(dataset_name=dataset_name, model_name=model_name)
    #     eval_path_2 = eval_path_template_2.format(dataset_name=dataset_name, model_name=model_name)
    #     false_rate_non_format, false_rate_format, false_format_nums, error_format_nums = claude3_check_dataset(eval_model=eval_model, results_file=save_path, save_path=eval_path_2, Ts=dataset_list[dataset_name][3], format_mode=False)
