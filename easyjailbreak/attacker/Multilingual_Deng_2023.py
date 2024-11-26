"""
Multilingual Class
============================================
This Class translates harmful queries from English into nine non-English
languages with varying levels of resources, and in intentional scenarios,
malicious users deliberately combine malicious instructions with multilingual 
prompts to attack LLMs.

Paper title: MULTILINGUAL JAILBREAK CHALLENGES IN LARGE LANGUAGE MODELS
arXiv Link: https://arxiv.org/pdf/2310.06474.pdf
Source repository: https://github.com/DAMO-NLP-SG/multilingual-safety-for-LLMs
"""
from collections import defaultdict
import requests
import logging
import json
logging.basicConfig(level=logging.INFO)
from easyjailbreak.metrics.Evaluator import EvaluatorGenerativeJudge, EvaluatorJudge
from easyjailbreak.attacker import AttackerBase
from easyjailbreak.datasets import JailbreakDataset, Instance
from easyjailbreak.mutation.rule import *
from collections import defaultdict

__all__ = ['Multilingual']
class Multilingual(AttackerBase):
    r"""
    Multilingual is a class for conducting jailbreak attacks on language models.
    It can translate harmful queries from English into nine non-English languages.
    """
    def __init__(self, attack_model, target_model, eval_model, eval_by_gpt4, multi_turn, jailbreak_datasets: JailbreakDataset):
        r"""
        Initialize the Multilingual attack instance.
        :param attack_model: The attack_model should be set to None.
        :param target_model: The target language model to be attacked.
        :param eval_model: The evaluation model to evaluate the attack results.
        :param jailbreak_datasets: The dataset to be attacked.
        """
        super().__init__(attack_model, target_model, eval_model, jailbreak_datasets)
        self.current_query: int = 0
        self.current_jailbreak: int = 0
        self.current_reject: int = 0
        self.multi_turn = multi_turn
        self.mutations = [
            # # Chinese
            # Translate(language='zh-CN'),
            # # Japanese
            # Translate(language='ja'),
            # # French
            # Translate(language='fr'),
            # # Arabic
            # Translate(language='ar'),
            # # Korean
            # Translate(language='ko'),
            # English
            Translate(language='en')
            ]
        if eval_by_gpt4:
            self.evaluator = EvaluatorJudge(eval_model)
        else:
            self.evaluator = EvaluatorJudge(eval_model, special_method="LlamaGuard")

    def single_attack(self, output_path, mode, instance: Instance) -> JailbreakDataset:
        r"""
        Execute the single attack process using provided prompts.
        """
        instance_dataset = JailbreakDataset([instance])
        mutated_instance_list = []
        updated_instance_list = []
        for mutation in self.mutations:
            transformed_dataset = mutation(instance_dataset)
            print(len(transformed_dataset))
            for item in transformed_dataset:
                mutated_instance_list.append(item)
        
        print(len(mutated_instance_list))
        print("****************")

        for instance in mutated_instance_list:
            if instance.jailbreak_prompt is not None:
                answer = self.target_model.generate(instance.jailbreak_prompt.format(translated_query = instance.translated_query))
            else:
                if mode == "":
                    if self.multi_turn:
                        print(instance.multi_turn_queries)
                        messages, answer_list = [], []
                        for query in instance.multi_turn_queries:
                            messages.append(query)
                            answer = self.target_model.generate(messages)
                            messages.append(answer)
                            answer_list.append(answer)
                    else:
                        print(instance.query)
                        answer = self.target_model.generate(instance.query)
                else:
                    print("this mode dosen't implement...")
                    exit()
            #en_answer = self.translate_to_en(answer)
            if self.multi_turn:
                en_answer = answer_list
            else:
                en_answer = answer
            instance.target_responses.append(en_answer)
            print(instance)
            self.write_to_json(data=instance.to_dict(), filename=output_path)
            print("****************")
            updated_instance_list.append(instance)

        return JailbreakDataset(updated_instance_list)
    
    def negative_sampling(self, instance: Instance) -> JailbreakDataset:
        r"""
        Execute the single attack process using provided prompts.
        """
        instance_dataset = JailbreakDataset([instance])
        mutated_instance_list = []
        updated_instance_list = []

        for mutation in self.mutations:
            transformed_dataset = mutation(instance_dataset)
            print(len(transformed_dataset))
            for item in transformed_dataset:
                mutated_instance_list.append(item)
        
        print(len(mutated_instance_list))
        print("****************")
 
        for instance in mutated_instance_list:
            print(instance.translated_query + "\n\n" + instance.reference_responses[0] + " Because")
            answer1, answer2, answer3 = self.target_model.completion(instance.translated_query + "\n\n" + instance.reference_responses[0] + " Because")
            # en_answer = self.translate_to_en(answer)
            instance.target_responses.append(answer1)
            instance.target_responses.append(answer2)
            instance.target_responses.append(answer3)
            self.write_to_json(data=instance.to_dict())
            print("****************")
            updated_instance_list.append(instance)

        return JailbreakDataset(updated_instance_list)

    def negative_sampling_v2(self, instance: Instance) -> JailbreakDataset:
        r"""
        Execute the single attack process using provided prompts.
        """
        updated_instance_list = []
        answer1, answer2, answer3 = self.target_model.completion(instance["query"] + "\n\n" + instance["reference_responses"][0])
        # en_answer = self.translate_to_en(answer)
        instance["target_responses"].append(answer1)
        instance["target_responses"].append(answer2)
        instance["target_responses"].append(answer3)
        self.write_to_json(data=instance)
        print("****************")
        updated_instance_list.append(instance)
        return JailbreakDataset(updated_instance_list)

    def write_to_json(self, data, filename="./text-davinci-003-saladbench.json"):
        with open(filename, 'a') as json_file:
            json.dump(data, json_file, indent=4)

    # /home/myt/Datasets/Eval_benchmark/HarmfulQA-SelfEval/HarmfulQA244-SelfEval-test.json
    # ./outputs_results/llm_response/HarmfulQA244/HarmfulQA244-chatgpt.json
    # ./outputs_results/llm_response/HarmfulQA244/HarmfulQA244-mistral-7b-sft-dpo-ICCmix-v1.json

    def read_json_tolist(self, filename="./outputs_results/llm_response/AdvBench520/AdvBench-claude3.json"):
        with open(filename, 'r') as file:
            data = json.load(file)
        my_list = []
        for item in data:
            instance = Instance(**item)
            my_list.append(instance)

        return JailbreakDataset(my_list)

    def attack(self, output_path = "", mode="TOP-ZS"):
        r"""
        Execute the attack process using Multilingual Jailbreak in Large Language Models.
        """
        logging.info("Jailbreak started!")
        self.attack_results = JailbreakDataset([])
        self.attack_results_list = []
        print(len(self.jailbreak_datasets))
        
        i=0
        try:
            for Instance in self.jailbreak_datasets:
                if i < 0:
                    i+=1
                    continue
                print("---------------------------")
                Instance['lang'] = "English"
                print(Instance)
                results = self.single_attack(output_path, mode, Instance)
                #results = self.negative_sampling(Instance)
                #results = self.negative_sampling_v2(Instance)
                for new_instance in results:
                    self.attack_results.add(new_instance)
                    self.attack_results_list.append(new_instance.to_dict())
            with open(output_path, 'w') as f:
                json.dump(self.attack_results_list, f)
        except KeyboardInterrupt:
            logging.info("Jailbreak interrupted by user!")
        #self.evaluator(self.attack_results)
        #self.update(self.attack_results)
        logging.info("Jailbreak finished!")

    def evaluation(self, input_path, eval_output_path):
        logging.info("load LLM outputs and start to evaluate!")
        self.attack_results = JailbreakDataset([])
        results = self.read_json_tolist(filename=input_path)
        print(len(results))
        i=0
        for new_instance in results:
            if i < 0:
                i+=1
                continue
            # if "unsafe" in new_instance.Evaluator_GenerativeJudge or "Unafe" in new_instance.Evaluator_GenerativeJudge:
            #     continue
            new_instance['lang'] = "English"
            self.attack_results.add(new_instance)
        print(len(self.attack_results))
        self.evaluator.eval_result_path = eval_output_path
        self.evaluator(self.attack_results)
        self.update(self.attack_results)
        self.log()
        logging.info("evaluation finished!")

    def update(self, dataset):
        r"""
        update the state of the Jailbroken based on the evaluation results of Datasets.
        """
        lang_groups = defaultdict(list)
        untranslate_query_groups = defaultdict(list)
        for prompt_node in dataset:
            self.current_jailbreak += prompt_node.num_jailbreak
            self.current_query += prompt_node.num_query
            self.current_reject += prompt_node.num_reject
            lang_groups[prompt_node.lang].append(prompt_node)
            untranslate_query_groups[prompt_node.query].append(prompt_node)

        self.lang_success_rates = {
            lang: sum(node.eval_results[-1] for node in group) / len(group) * 100
            for lang, group in lang_groups.items()
        }

        self.query_success_rates = {
            query: sum(node.eval_results[-1] for node in group) / len(group) * 100
            for query, group in untranslate_query_groups.items()
        }

        self.total_success_rate = sum(node.eval_results[-1] for node in dataset) / len(dataset) * 100

    def log(self):
        r"""
        report the attack results.
        """
        logging.info("======Jailbreak report:======")
        logging.info(f"Total queries: {self.current_query}")
        logging.info(f"Total jailbreak: {self.current_jailbreak}")
        logging.info(f"Total reject: {self.current_reject}")

        for lang, rate in self.lang_success_rates.items():
            logging.info(f"Success rate of {lang}: {rate:.2f}%")

        for query, rate in self.query_success_rates.items():
            logging.info(f"Jailbreak success rate for query '{query}': {rate:.2f}%")

        logging.info(f"Total success rate: {self.total_success_rate:.2f}%")
        logging.info("========Report End===========")
    
    def translate_to_en(self, text, src_lang='auto'):
        r"""
        Translate target response to English.
        """
        googleapis_url = 'https://translate.googleapis.com/translate_a/single'
        url = '%s?client=gtx&sl=%s&tl=%s&dt=t&q=%s' % (googleapis_url,src_lang,'en',text)
        data = requests.get(url).json()
        res = ''.join([s[0] for s in data[0]])
        return res
