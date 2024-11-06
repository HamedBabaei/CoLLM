# -*- coding: utf-8 -*-
import argparse
import json
import os
import time

import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoModelForSeq2SeqLM,
                          AutoTokenizer, BloomForCausalLM, BloomTokenizerFast)

_ = load_dotenv(find_dotenv())


def read_json_file(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def save_json_file(data, file_path, indent=4):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=indent)


def evaluate(actual, predicted):
    clf_report_dict = classification_report(
        actual, predicted, digits=4, output_dict=True
    )
    return clf_report_dict


class OpenAIModel:
    def __init__(self, model_path, prompt_template, gpt_max_tokens=2):
        self.client = OpenAI(api_key=os.environ["OPENAI_KEY"])
        self.prompt_template = prompt_template
        self.model_path = model_path
        self.gpt_max_tokens = gpt_max_tokens

    def check_all_is_done(self, results):
        is_done = True
        for result in results:
            if result["check"] == False:
                is_done = False
        return is_done

    def make_prediction(self, data):
        prompt = self.prompt_template.replace("[A]", data["text_a"]).replace(
            "[B]", data["text_b"]
        )
        print(prompt)
        messages = [{"role": "user", "content": prompt}]
        if "o1-preview" in self.model_path:
            response = self.client.chat.completions.create(
                model=self.model_path,
                messages=messages,
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model_path,
                messages=messages,
                temperature=0,
                max_tokens=self.gpt_max_tokens,
            )
        result = {"data": data, "prompt": prompt, "response": response}
        return result

    def test(self, X, generative_test=False):
        results = []
        for index, data in enumerate(X):
            results.append({"check": False})

        assert self.check_all_is_done(results) == False

        while not self.check_all_is_done(results):
            for index, data in tqdm(enumerate(X)):
                if results[index]["check"] != True:
                    try:
                        results[index]["result"] = self.make_prediction(data=data)
                        print(
                            results[index]["result"]["response"]
                            .choices[0]
                            .message.content.lower()
                        )
                        results[index]["check"] = True
                    except Exception as err:
                        print(f"UNexpected {err}, {type(err)}")
                        print("Going to sleep for 5 second!")
                        time.sleep(5)
        predicts = [
            result["result"]["response"].choices[0].message.content.lower()
            for result in results
        ]
        labels = [result["result"]["data"]["label"].lower() for result in results]
        return predicts, labels


class LLMClassifierModel:
    def __init__(self, model_path, task, prompt_template, max_new_tokens=5):
        self.prompt_template = prompt_template
        self.device = "cuda"
        self.task = task
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        if "flan-t5" in model_path:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path, device_map="balanced"
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, token=os.environ["HUGGINGFACE_ACCESS_TOKEN"]
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="balanced",
                token=os.environ["HUGGINGFACE_ACCESS_TOKEN"],
            )
        # self.tokenizer.pad_token = self.tokenizer.bos_token
        if "llama" in model_path:
            eot = "<|eot_id|>"
            eot_id = self.tokenizer.convert_tokens_to_ids(eot)
            self.tokenizer.pad_token = eot
            self.tokenizer.pad_token_id = eot_id
            self.model.resize_token_embeddings(len(self.tokenizer))
        elif "flan-t5" in model_path:
            pass
        else:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.ANSWER_SET = {
            "correct": [
                "yes",
                "true",
                "correct",
                "right",
                "good",
                "accurate",
                "valid",
                "exact",
                "appropriate",
                "proper",
                "precise",
            ],
            "incorrect": [
                "no",
                "false",
                "incorrect",
                "wrong",
                "bad",
                "inaccurate",
                "mistaken",
                "invalid",
                "improper",
                "untrue",
            ],
        }

        self.index2label = {0: "correct", 1: "incorrect"}
        self.label2index = [
            self.tokenizer("correct").input_ids[-1],
            self.tokenizer("incorrect").input_ids[-1],
        ]
        self.answer_sets_token_id = {}
        for label, answer_set in self.ANSWER_SET.items():
            self.answer_sets_token_id[label] = []
            for answer in answer_set:
                if self.check_answer_set_tokenizer(answer):
                    # print(answer, '  ', self.tokenizer(answer).input_ids)
                    if "flan-t5" in model_path:
                        self.answer_sets_token_id[label].append(
                            self.tokenizer(answer).input_ids[0]
                        )
                    else:
                        self.answer_sets_token_id[label].append(
                            self.tokenizer(answer).input_ids[-1]
                        )

    def check_answer_set_tokenizer(self, answer):
        return len(self.tokenizer(answer).input_ids) == 2

    def get_probas_yes_no(self, outputs):
        probas_yes_no = outputs.scores[0][
            :,
            self.answer_sets_token_id["correct"]
            + self.answer_sets_token_id["incorrect"],
        ].softmax(-1)
        return probas_yes_no

    def generate_for_llm(self, tokenized_input_data):
        with torch.no_grad():
            outputs = self.model.generate(
                **tokenized_input_data,
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=self.max_new_tokens,
                # do_sample=False,
                num_beams=10,
                output_scores=True,
                return_dict_in_generate=True,
            )
        return outputs

    def generate_for_one_input(self, tokenized_input_data):
        outputs = self.generate_for_llm(tokenized_input_data=tokenized_input_data)
        probas_yes_no = self.get_probas_yes_no(outputs=outputs)
        yes_probas = probas_yes_no[:, : len(self.ANSWER_SET["correct"])].sum(dim=1)
        no_proba = probas_yes_no[:, len(self.ANSWER_SET["correct"]) :].sum(dim=1)
        probas = torch.cat((yes_probas.reshape(-1, 1), no_proba.reshape(-1, 1)), -1)
        probas_per_candidate_tokens = torch.max(probas, dim=1)
        sequence_probas = [float(proba) for proba in probas_per_candidate_tokens.values]
        sequences = [
            self.index2label[int(indice)]
            for indice in probas_per_candidate_tokens.indices
        ]
        # if probas[0][0] >= 0.6:
        #     return ['correct']
        # else:
        #     return ['incorrect']
        return sequences

    def generate_for_multiple_input(self, tokenized_input_data):
        return self.generate_for_one_input(tokenized_input_data=tokenized_input_data)

    def generate(self, input_data, generative_test):
        tokenized_input_data = self.tokenizer(input_data, return_tensors="pt")
        tokenized_input_data.to(self.device)
        if generative_test:
            with torch.no_grad():
                outputs = self.model.generate(
                    **tokenized_input_data,
                    pad_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=self.max_new_tokens,
                )
            generated_texts = self.tokenizer.batch_decode(
                outputs.cpu(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            generated_texts[0] = generated_texts[0][len(input_data[0]) :]
        else:
            if len(input_data) == 1:
                generated_texts = self.generate_for_one_input(
                    tokenized_input_data=tokenized_input_data
                )
            else:
                generated_texts = self.generate_for_multiple_input(
                    tokenized_input_data=tokenized_input_data
                )
        return generated_texts

    def test(self, X, generative_test=False):
        predicts, labels = [], []
        for index, data in tqdm(enumerate(X)):
            if self.task == "C":
                prompt = (
                    self.prompt_template.replace("[H]", data["h"].lower())
                    .replace("[T]", data["t"].lower())
                    .replace("[R]", data["r"].replace("_", " "))
                )
            else:
                prompt = self.prompt_template.replace("[A]", data["text_a"]).replace(
                    "[B]", data["text_b"]
                )

            predict = self.generate(
                input_data=[prompt], generative_test=generative_test
            )
            predicts.append(predict)
            labels.append(data["label"])

        return predicts, labels


class InferenceFactory:
    def __init__(self) -> None:
        self.model_path = {
            "gpt4_0613": "gpt-4-0613",
            "gpt4_1106_preview": "gpt-4-1106-preview",
            "gpt4_0125_preview": "gpt-4-0125-preview",
            "gpt4_turbo_2024_04_09": "gpt-4-turbo-2024-04-09",
            "llama3": "meta-llama/Llama-3.1-8B",  # "meta-llama/Llama-3.1-8B",
            "mistral": "mistralai/Mistral-7B-v0.3",
            "vicuna": "lmsys/vicuna-7b-v1.5",
            "o1preview": "o1-preview",
            "flan_t5_small": "google/flan-t5-small",
            "flan_t5_base": "google/flan-t5-base",
            "flan_t5_large": "google/flan-t5-large",
            "flan_t5_xl": "google/flan-t5-xl",
            "flan_t5_xxl": "google/flan-t5-xxl",
        }

    def __call__(self, model_name, prompt_template, task):
        if "gpt4" in model_name or "o1preview" in model_name:
            model = OpenAIModel(
                model_path=self.model_path[model_name], prompt_template=prompt_template
            )
            return model
        else:
            model = LLMClassifierModel(
                task=task,
                model_path=self.model_path[model_name],
                prompt_template=prompt_template,
            )
            return model
        print(
            f"Oops! That was not valid model name. Valid models are: {list(self.models.keys())}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run_no", required=True)
    parser.add_argument("--rq", required=True)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()

    if args.task == "B":
        template = "Identify whether the following statement is true or false: \n\nStatement: [B] is a subtype of [A]. \nThis statement is a: "
    else:
        template = "Identify whether the following statement is true or false: \n[H] is [R] [T]. Answer:"

    dataset_path_dict = {
        "umls-B": "dataset/LLMs4OL-2023-paper-Task-B-UMLS-test.json",
        "schema-B": "dataset/LLMs4OL-2023-paper-Task-B-schemaorg-test.json",
        "umls-C": "dataset/LLMs4OL-2023-paper-Task-C-UMLS-test.json",
    }

    test_path = dataset_path_dict.get(f"{args.dataset}-{args.task}")
    dataset = read_json_file(test_path)
    label_mapper = {
        "correct": [
            "yes",
            "true",
            "correct",
            "right",
            "good",
            "accurate",
            "valid",
            "exact",
            "appropriate",
            "proper",
            "precise",
        ],
        "incorrect": [
            "no",
            "false",
            "incorrect",
            "wrong",
            "bad",
            "inaccurate",
            "mistaken",
            "invalid",
            "improper",
            "untrue",
        ],
    }
    inference_model = InferenceFactory()(
        model_name=args.model_name, task=args.task, prompt_template=template
    )
    if args.model_name in ["llama3", "mistral", "vicuna"] and args.task == "C":
        predicts, labels = inference_model.test(dataset, generative_test=True)
    else:
        predicts, labels = inference_model.test(dataset, generative_test=False)

    predictions = []
    for predict in predicts:
        predict_label = "incorrect"
        for label in label_mapper["correct"]:
            if label in predict[0].lower():
                predict_label = "correct"
        predictions.append(predict_label)

    eval_results = evaluate(labels, predictions)

    results_dict = {
        "args": vars(args),
        "task": args.task,
        "model_name": args.model_name,
        "result": eval_results,
        "predictions": predictions,
        "labels": labels,
    }
    save_json_file(
        data=results_dict,
        file_path=f"results/Task{args.task}-{args.dataset.upper()}-{args.model_name}-{args.rq}-{args.run_no}.json",
    )
