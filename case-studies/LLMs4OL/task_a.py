# -*- coding: utf-8 -*-
import argparse
import json
import os
import time

import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (AutoTokenizer, BloomForCausalLM, BloomTokenizerFast,
                          LlamaForCausalLM)

_ = load_dotenv(find_dotenv())


def read_json_file(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def save_json_file(data, file_path, indent=4):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=indent)


class BiomedicalDataset(Dataset):
    def __init__(self, data, template, label_mapper, is_train=False):
        self.template = template
        self.is_train = is_train
        self.label_mapper = label_mapper
        self.use_sentence = True if "[SENTENCE]" in self.template else False
        self.data = data
        print(
            f"BiomedicalDataset:{'Train-SET' if self.is_train else 'Test-SET'}--- {template}: {self.template}"
        )

    def __len__(self):
        return len(self.data)

    def __repr__(self) -> str:
        return f"BiomedicalDataset:{'Train-SET' if self.is_train else 'Test-SET'}--- Template-in-use: {self.template}"

    def __getitem__(self, index):
        item = self.data[index]
        concept = str(item["concept"]).lower()
        sample = self.template.replace("[A]", concept)
        labels = []
        for label in eval(item["label-str"]):
            for lab in self.label_mapper[label]:
                labels.append(lab)
        label = list(set(labels))
        label = [lab.lower() for lab in label]
        if self.use_sentence:
            sample = sample.replace("[SENTENCE]", concept)
        if self.is_train:
            sample = sample  # .replace("[MASK]", 'or '.join(item['label-names']))
        return {"sample": sample, "label": label}

    def collate_fn(self, batchs):
        batchs_clear = {"sample": [], "label": []}
        for batch in batchs:
            batchs_clear["sample"].append(batch["sample"])
            batchs_clear["label"].append(batch["label"])
        return batchs_clear


class BaseLM:
    def __init__(self, model_path, device, top_n) -> None:
        self.tokenizer = None
        self.model = None
        self.device = device
        self.top_n = top_n
        self.model_path = model_path
        pass

    def load(self):
        pass

    def make_batch_prediction(self, Xs: list):
        pass

    def batch_tokenize(self, Xs):
        inputs = self.tokenizer(Xs, return_tensors="pt", padding=True)
        inputs.to(self.device)
        return inputs

    def single_tokenize(self, X):
        inputs = self.tokenizer(X, return_tensors="pt")
        inputs.to(self.device)
        return inputs

    def output_cleaner(self, pred, **kwargs):
        return pred

    def predict(self, X: str):
        pass


class EncoderDecoderLM(BaseLM):
    def __init__(self, model_path, device, top_n) -> None:
        super().__init__(model_path, device, top_n)

    def predict(self, X: str, max_new_tokens: int = 5):
        inputs = self.single_tokenize(X)
        with torch.no_grad():
            sequence_ids = self.model.generate(
                inputs.input_ids,
                num_beams=50,
                num_return_sequences=self.top_n,
                max_new_tokens=max_new_tokens,
            )
        sequences = self.tokenizer.batch_decode(sequence_ids, skip_special_tokens=True)
        sequences = [self.output_cleaner(seq, prompt=X) for seq in sequences]
        logits = [0 for seq in sequences]
        return sequences, logits

    def make_batch_prediction(self, Xs, max_new_tokens=5):
        predictions, logits = [], []
        inputs = self.batch_tokenize(Xs)
        with torch.no_grad():
            sequence_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
            )
        sequences = self.tokenizer.batch_decode(
            sequence_ids.cpu(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        sequences_logist = [0 for _ in sequences]
        for index in range(0, len(Xs)):
            started_idx = self.top_n * index
            end_idx = self.top_n * (index + 1)
            predictions.append(sequences[started_idx:end_idx])
            logits.append(sequences_logist[started_idx:end_idx])
        predictions = []
        for predicts, prompt in zip(predictions, Xs):
            prediction = []
            for predict in predicts:
                prediction.append(self.output_cleaner(predict, prompt=prompt))
            predictions.append(prediction)
        return predictions, logits

    def make_single_batch_prediction(self, Xs):
        predictions, logits = [], []
        for X in Xs:
            predict, logit = self.predict(X)
            predictions.append(predict)
            logits.append(logit)
        return predictions, logits


class BLOOMDecoderLM(EncoderDecoderLM):
    def __init__(self, model_path, device, top_n) -> None:
        super().__init__(model_path, device, top_n)

    def load(self):
        self.tokenizer = BloomTokenizerFast.from_pretrained(self.model_path)
        self.model = BloomForCausalLM.from_pretrained(self.model_path)
        print(f"Loaded BloomForCausalLM from{self.model_path}")
        self.model.to(self.device)
        self.model.eval()

    def output_cleaner(self, pred, **kwargs):
        pred = pred.replace(kwargs["prompt"], "")
        return pred.replace("<pad>", "").replace("</s>", "").strip()


class LLaMADecoderLM(EncoderDecoderLM):
    def __init__(self, model_path, device, top_n) -> None:
        super().__init__(model_path, device, top_n)

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            token=os.environ["HUGGINGFACE_ACCESS_TOKEN"],
            padding_side="left",
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = LlamaForCausalLM.from_pretrained(
            self.model_path,
            # load_in_8bit=False,
            # torch_dtype=torch.float16,
            token=os.environ["HUGGINGFACE_ACCESS_TOKEN"],
            device_map="auto",
        )
        print(f"Loaded LLamaForCausalLM from{self.model_path}")
        # self.model.to(self.device)
        # self.model.eval()

    def output_cleaner(self, pred, **kwargs):
        pred = pred.replace(kwargs["prompt"], "")
        return pred


class InferenceFactory:
    def __init__(self) -> None:
        self.models = (BLOOMDecoderLM,)
        self.model_path = {
            "bloom_560m": "bigscience/bloom-560m",
            "bloom_1b1": "bigscience/bloom-1b1",
            "bloom_1b7": "bigscience/bloom-1b7",
            "bloom_3b": "bigscience/bloom-3b",
            "bloom_7b1": "bigscience/bloom-7b1",
            "llama3": "meta-llama/Llama-3.1-8B",
        }

    def __call__(self, model_name, device, top_n=1):
        try:
            if model_name == "llama3":
                model_module = LLaMADecoderLM
            else:
                model_module = BLOOMDecoderLM
            model = model_module(self.model_path[model_name], device, top_n)
            model.load()
            return model
        except ValueError:
            print(
                f"Oops! That was not valid model name. Valid models are: {list(self.models.keys())}"
            )


def precision_at_k(actual, predicted):
    act_set = set(actual)
    pred_set = set(predicted)
    result = len(act_set & pred_set) / float(len(predicted))
    return result * 100


def apk(actual, predicted, k):
    if not actual:
        return 0.0
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        # first condition checks whether it is valid prediction
        # second condition checks if prediction is not repeated
        if p in actual and p.lower() not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / min(len(actual), k)


def apk2(actual, predicted, k):
    if not actual:
        return 0.0
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        # first condition checks whether it is valid prediction
        # second condition checks if prediction is not repeated
        itis = False
        for act in actual:
            if act.lower() in p.lower():
                itis = True
        if itis and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
        # if p in actual and p.lower() not in predicted[:i]:
        #     num_hits += 1.0
        #     score += num_hits / (i+1.0)
    return score / min(len(actual), k)


def mapk(actual, predicted, k):
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def mapk2(actual, predicted, k):
    return np.mean([apk2(a, p, k) for a, p in zip(actual, predicted)])


class EvaluationMetrics:
    def __init__(self, ks: list, metric="map") -> None:
        self.ks = ks
        self.metric = metric

    def evaluate(self, actual: list, predicted: list):
        if self.metric == "map":
            return self.MAP(actual, predicted)
        else:
            return self.AP(actual, predicted)

    def MAP(self, actual: list, predicted: list):
        results_dict = {}
        for k in self.ks:
            results_dict["MAP@" + str(k)] = mapk(
                actual=actual, predicted=predicted, k=k
            )
        for k in self.ks:
            results_dict["MAP-V2@" + str(k)] = mapk2(
                actual=actual, predicted=predicted, k=k
            )
        return results_dict

    def AP(self, actual: list, predicted: list):
        results_dict = {}
        for k in self.ks:
            results_dict["AP@" + str(k)] = [
                apk(actual=actual, predicted=predicted, k=k)
                for a, p in zip(actual, predicted)
            ]
        return results_dict


def inference(model, dataloader, max_new_tokens: int = 5):
    predictions, logits, labels = [], [], []
    for batch in tqdm(dataloader):
        prediction, logit = model.make_batch_prediction(
            list(batch["sample"]), max_new_tokens=max_new_tokens
        )
        for pred, log, label in zip(prediction, logit, list(batch["label"])):
            predictions.append(pred)
            logits.append(list(log))
            labels.append(label)
    return predictions, logits, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--run_no", required=True)
    parser.add_argument("--rq", required=True)
    args = parser.parse_args()

    daaset_name = "snomedct_us"
    if "llama3" == args.model_name:
        template = "Identify which category '[A]' belongs to in biomedicine domain. \n Term: '[A]' \nType:"
        max_new_tokens = 10
    else:
        template = (
            "Perform a sentence completion on the following sentence:\n"
            "Sentence: [SENTENCE]. [A] in biomedicine is a"
        )
        max_new_tokens = 5

    device = "cuda"
    batch_size = 128
    test_path = "dataset/LLMs4OL-2023-paper-Task-A-snomedct_us-test.json"

    label_mapper_path = (
        "dataset/LLMs4OL-2023-paper-Task-A-snomedct_us-label-mapper.json"
    )
    dataset = read_json_file(test_path)
    label_mapper = read_json_file(label_mapper_path)

    test_dataset = BiomedicalDataset(
        data=dataset, template=template, label_mapper=label_mapper
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=test_dataset.collate_fn,
    )
    inference_model = InferenceFactory()(model_name=args.model_name, device=device)
    predictions, logits, labels = inference(
        model=inference_model, dataloader=test_dataloader, max_new_tokens=max_new_tokens
    )
    evaluator = EvaluationMetrics(ks=[1, 5, 10], metric="map")
    results = evaluator.evaluate(actual=labels, predicted=predictions)
    results_dict = {
        "args": vars(args),
        "task": "A",
        "model_name": args.model_name,
        "result": results,
        "predictions": predictions,
        "labels": labels,
        # "logits": logits
    }
    save_json_file(
        data=results_dict,
        file_path=f"results/TaskA-{daaset_name.upper()}-{args.model_name}-{args.rq}-{args.run_no}.json",
    )
