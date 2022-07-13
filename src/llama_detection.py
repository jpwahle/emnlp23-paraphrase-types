# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import argparse
import json
import os
import time
from typing import List

import fire
import numpy as np
from datasets import load_dataset
from sklearn.metrics import f1_score
from tqdm import tqdm

from llama import Llama

ALL_TYPES = {
    "Derivational Changes",
    "Inflectional Changes",
    "Modal Verb Changes",
    "Spelling changes",
    "Change of format",
    "Same Polarity Substitution (contextual)",
    "Same Polarity Substitution (habitual)",
    "Same Polarity Substitution (named ent.)",
    "Converse substitution",
    "Opposite polarity substitution (contextual)",
    "Opposite polarity substitution (habitual)",
    "Synthetic/analytic substitution",
    "Coordination changes",
    "Diathesis alternation",
    "Ellipsis",
    "Negation switching",
    "Subordination and nesting changes",
    "Direct/indirect style alternations",
    "Punctuation changes",
    "Syntax/discourse structure changes",
    "Entailment",
    "Identity",
    "Non-paraphrase",
    "Addition/Deletion",
    "Change of order",
    "Semantic-based",
}


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


def classify(
    data, model, max_gen_len, temperature, top_p, max_batch_size, num_examples=100
):
    """Classifies the data using the provided model and parameters. Now forwards max_batch_size prompts into the model instead of one.

    Args:
        data (list): The data to classify.
        model (object): The model to use for classification.
        max_gen_len (int): The maximum generation length.
        temperature (float): The temperature parameter for the model.
        top_p (float): The top_p parameter for the model.
        max_batch_size (int): The maximum batch size for the model.
        num_examples (int, optional): The number of examples to classify. Defaults to 100.

    Returns:
        tuple: The true and predicted labels.
    """
    y_true = []
    y_pred = []

    for i in tqdm(range(0, num_examples, max_batch_size)):
        batch = data[i : i + max_batch_size]
        user_messages = [instance["messages"][0]["content"] for instance in batch]
        true_response_labels = [
            set(instance["messages"][1]["content"].split(", ")) for instance in batch
        ]

        # Call the API and retry if it fails

        results = model.text_completion(
            user_messages,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        predicted_response_labels = [
            set(result["generation"].split(", ")) for result in results
        ]

        if results:
            y_true.extend(true_response_labels)
            y_pred.extend(predicted_response_labels)

    return y_true, y_pred


def evaluate(y_true, y_pred):
    """Evaluates the performance of a classification model.

    Args:
        y_true (list): The true labels.
        y_pred (list): The predicted labels.

    Returns:
        tuple: A tuple containing the F1 score and accuracy.

    Example:
        ```python
        y_true = [[1, 0, 1], [0, 1, 0]]
        y_pred = [[1, 1, 0], [0, 1, 1]]

        f1, acc = evaluate(y_true, y_pred)
        print(f1)
        print(acc)
        ```
    """

    y_true_bin = [[1 if t in labels else 0 for t in ALL_TYPES] for labels in y_true]
    y_pred_bin = [[1 if t in labels else 0 for t in ALL_TYPES] for labels in y_pred]

    # Compute F1 score
    f1 = f1_score(y_true_bin, y_pred_bin, average="micro")

    # Convert lists to numpy arrays for easier calculations
    y_true_np = np.array(y_true_bin)
    y_pred_np = np.array(y_pred_bin)

    # Calculate per-class accuracy
    acc = np.mean(np.equal(y_true_np, y_pred_np).astype(int))

    return f1, acc


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
    y_true, y_pred = classify(
        test_data,
        model,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
        max_batch_size=max_batch_size,
        num_examples=num_examples,
    )
    f1, acc = evaluate(y_true, y_pred)

    print(f"Model: {ckpt_dir}")
    print(f"Eval set size: {len(y_pred)}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Accuracy: {acc:.2f}")


if __name__ == "__main__":
    fire.Fire(main)
