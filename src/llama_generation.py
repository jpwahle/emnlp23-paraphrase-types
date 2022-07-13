import argparse
import json
import os
import time
from typing import List

import fire
import numpy as np
from datasets import load_dataset
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge import Rouge
from sklearn.metrics import f1_score
from tqdm import tqdm

from llama import Llama

# Install required libraries first
# pip install rouge nltk sacrebleu


# Load data
def load_data(filename):
    """Loads data from a file in JSON format.

    Args:
        filename (str): The path to the file to load.

    Returns:
        list: The loaded data as a list of dictionaries.

    Example:
        ```python
        filename = "data.json"
        data = load_data(filename)
        print(data)
        ```
    """
    with open(filename, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f.readlines()]
    return data


def generate_paraphrases(
    data, model, max_gen_len, temperature, top_p, max_batch_size, num_examples=3
):
    paraphrases = []

    for i in tqdm(range(0, num_examples, max_batch_size)):
        batch = data[i : i + max_batch_size]
        user_messages = [instance["messages"][0]["content"] for instance in batch]

        results = model.text_completion(
            user_messages,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        paraphrases.extend([result["generation"] for result in results])

    return paraphrases


def evaluate(paraphrases, references):
    """
    Evaluates the quality of paraphrases compared to reference texts.

    Args:
        paraphrases (list): The generated paraphrases.
        references (list): The reference texts.

    Returns:
        dict: A dictionary containing the evaluation scores.

    Example:
        ```python
        paraphrases = ["Paraphrase 1", "Paraphrase 2"]
        references = ["Reference 1", "Reference 2"]

        scores = evaluate(paraphrases, references)
        print(scores)
        ```
    """

    rouge = Rouge()

    # ROUGE scores
    rouge_scores = rouge.get_scores(paraphrases, references, avg=True)

    # BLEU scores
    smoothie = SmoothingFunction().method4
    bleu_scores = [
        sentence_bleu([ref], paraphrase, smoothing_function=smoothie)
        for ref, paraphrase in zip(references, paraphrases)
    ]
    avg_bleu = sum(bleu_scores) / len(bleu_scores)

    return {
        "ROUGE-1": rouge_scores["rouge-1"]["f"],
        "ROUGE-2": rouge_scores["rouge-2"]["f"],
        "ROUGE-L": rouge_scores["rouge-l"]["f"],
        "BLEU": avg_bleu,
    }


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    data_file: str,
    num_examples: int = 1000,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 2048,
    max_gen_len: int = 1024,
    max_batch_size: int = 4,
):
    model = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # Load data and predict
    test_data = load_data(data_file)
    generated_paraphrases = generate_paraphrases(
        test_data,
        model,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
        max_batch_size=max_batch_size,
        num_examples=num_examples,
    )
    references = [item["messages"][1]["content"] for item in test_data[:num_examples]]
    scores = evaluate(generated_paraphrases, references)
    print(f"Model: {ckpt_dir}")
    print(scores)


if __name__ == "__main__":
    fire.Fire(main)
