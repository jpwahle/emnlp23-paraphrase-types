import argparse
import xml.etree.ElementTree as ET

import evaluate
import numpy as np
import pandas as pd
import requests
import spacy
from datasets import concatenate_datasets, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

nlp = spacy.load("en_core_web_sm")

# Groups
grouped_types = {
    "Morphology-based changes": [
        "Inflectional changes",
        "Modal verb changes",
        "Derivational changes",
    ],
    "Lexicon-based changes": [
        "Spelling changes",
        "Change of format",
        "Same Polarity Substitution (contextual)",
        "Same Polarity Substitution (habitual)",
        "Same Polarity Substitution (named ent.)",
    ],
    "Lexico-syntactic based changes": [
        "Converse substitution",
        "Opposite polarity substitution (contextual)",
        "Opposite polarity substitution (habitual)",
        "Synthetic/analytic substitution",
    ],
    "Syntax-based changes": [
        "Coordination changes",
        "Diathesis alternation",
        "Ellipsis",
        "Negation switching",
        "Subordination and nesting changes",
    ],
    "Discourse-based changes": [
        "Direct/indirect style alternations",
        "Punctuation changes",
        "Syntax/discourse structure changes",
    ],
    "Extremes": ["Entailment", "Identity", "Non-paraphrase"],
    "Others": ["Addition/Deletion", "Change of order", "Semantic-based"],
}


def create_label_maps(etpc):
    """
    Creates label maps for the ETPC paraphrase types.

    Returns:
        tuple: A tuple containing the following dictionaries:
            - label2cls_id: A dictionary mapping paraphrase types to class IDs.
            - cls_id2label: A dictionary mapping class IDs to paraphrase types.
            - paraphrase_type2cls_id: A dictionary mapping paraphrase types to class IDs.
            - paraphrase_id2cls_type: A dictionary mapping class IDs to paraphrase types.
            - paraphrase_type_to_category: A dictionary mapping paraphrase types to categories.
            - cls_id2paraphrase_type_id: A dictionary mapping class IDs to paraphrase type IDs.
            - paraphrase_type_id2cls_id: A dictionary mapping paraphrase type IDs to class IDs.

    Example:
        ```python
        label_maps = create_label_maps(etpc)
        print(label_maps)
        ```"""
    # Flatten paraphrase_types as list
    all_types = {el for sublist in etpc["paraphrase_types"] for el in sublist}

    # Download xml with paraphrase types to ids from url https://github.com/venelink/ETPC/blob/master/Corpus/paraphrase_types.xml
    url = "https://raw.githubusercontent.com/venelink/ETPC/master/Corpus/paraphrase_types.xml"
    r = requests.get(url)
    root = ET.fromstring(r.text)

    # Get paraphrase types, ids and categories
    paraphrase_types = [child.find("type_name").text for child in root]
    paraphrase_type_ids = [int(child.find("type_id").text) for child in root]
    paraphrase_type_categories = [child.find("type_category").text for child in root]

    # Create dictionary with paraphrase type as key and paraphrase type id as value
    paraphrase_type2cls_id = dict(zip(paraphrase_types, paraphrase_type_ids))
    paraphrase_id2cls_type = dict(zip(paraphrase_type_ids, paraphrase_types))

    # Create dictionary with paraphrase type as key and paraphrase type category as value
    paraphrase_type_to_category = dict(
        zip(paraphrase_types, paraphrase_type_categories)
    )

    # Add 0 for no paraphrase to all dictionaries
    paraphrase_type2cls_id["no_paraphrase"] = 0
    paraphrase_id2cls_type[0] = "no_paraphrase"
    paraphrase_type_to_category["no_paraphrase"] = "no_paraphrase"

    # Create label2id and id2label for etpc paraphrase_types
    label2cls_id = {label: i + 1 for i, label in enumerate(all_types)}
    cls_id2label = {i: label for label, i in label2cls_id.items()}

    # Add 0 for no paraphrase to all dictionaries
    label2cls_id["no_paraphrase"] = 0
    cls_id2label[0] = "no_paraphrase"

    # Create a map from ids to the ones in paraphrase_type_to_id and vice versa
    cls_id2paraphrase_type_id = {
        i: paraphrase_type2cls_id[cls_id2label[i]] for i in cls_id2label
    }
    paraphrase_type_id2cls_id = {
        paraphrase_type2cls_id[cls_id2label[i]]: i for i in cls_id2label
    }

    # Create a dictionary that maps ids from label2cls_id to the ones in paraphrase_type_to_id using the type label and vice versa
    cls_id2paraphrase_type_id = {
        i: paraphrase_type2cls_id[cls_id2label[i]] for i in cls_id2label
    }
    paraphrase_type_id2cls_id = {
        paraphrase_type2cls_id[cls_id2label[i]]: i for i in cls_id2label
    }

    return (
        label2cls_id,
        cls_id2label,
        paraphrase_type2cls_id,
        paraphrase_id2cls_type,
        paraphrase_type_to_category,
        cls_id2paraphrase_type_id,
        paraphrase_type_id2cls_id,
    )


def tokenize_and_align_labels(
    examples,
    sentence1_key,
    sentence2_key,
    paraphrase_type_id2cls_id,
    tokenizer,
):
    """
    Tokenizes the input sentences and aligns the labels with the tokenized inputs.

    Args:
        examples (dict): The input examples.
        sentence1_key (str): The key for the first sentence in the examples.
        sentence2_key (str): The key for the second sentence in the examples. Can be None.
        paraphrase_type_id2cls_id (dict): A dictionary mapping paraphrase type IDs to class IDs.
        tokenizer: The tokenizer object used for tokenization.

    Returns:
        dict: A dictionary containing the tokenized inputs with aligned labels.

    Example:
        ```python
        examples = {
            "sentence1": "This is sentence 1.",
            "sentence2": "This is sentence 2.",
            "sentence1_segment_location": [0, 1, 1, 2, 2],
            "sentence2_segment_location": [0, 1, 1, 2, 2],
        }
        sentence1_key = "sentence1"
        sentence2_key = "sentence2"
        paraphrase_type_id2cls_id = {0: 0, 1: 1, 2: 2}
        tokenizer = Tokenizer()

        tokenized_inputs = tokenize_and_align_labels(
            examples, sentence1_key, sentence2_key, paraphrase_type_id2cls_id, tokenizer
        )
        print(tokenized_inputs)
        ```"""
    sentence1_key = sentence1_key + "_tokenized"
    sentence2_key = sentence2_key + "_tokenized"

    args = (
        (examples[sentence1_key],)
        if sentence2_key is None
        else (examples[sentence1_key], examples[sentence2_key])
    )
    tokenized_inputs = tokenizer(*args, truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(
        zip(
            examples["sentence1_segment_location"],
            examples["sentence2_segment_location"],
        )
    ):
        # Map all labels to the id using paraphrase_
        label = [
            paraphrase_type_id2cls_id[paraphrase_id]
            for paraphrase_id in label[0] + label[1]
        ]

        word_ids = tokenized_inputs.word_ids(
            batch_index=i
        )  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None or word_idx == previous_word_idx:
                label_ids.append(-100)
            else:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def split_dataset_binary(dataset, seed=42):
    """
    Tokenizes the input sentences and aligns the labels with the tokenized inputs.

    Args:
        examples (dict): The input examples.
        sentence1_key (str): The key for the first sentence in the examples.
        sentence2_key (str): The key for the second sentence in the examples. Can be None.
        paraphrase_type_id2cls_id (dict): A dictionary mapping paraphrase type IDs to class IDs.
        tokenizer: The tokenizer object used for tokenization.

    Returns:
        dict: A dictionary containing the tokenized inputs with aligned labels.

    Example:
        ```python
        examples = {
            "sentence1": "This is sentence 1.",
            "sentence2": "This is sentence 2.",
            "sentence1_segment_location": [0, 1, 1, 2, 2],
            "sentence2_segment_location": [0, 1, 1, 2, 2],
        }
        sentence1_key = "sentence1"
        sentence2_key = "sentence2"
        paraphrase_type_id2cls_id = {0: 0, 1: 1, 2: 2}
        tokenizer = Tokenizer()

        tokenized_inputs = tokenize_and_align_labels(
            examples, sentence1_key, sentence2_key, paraphrase_type_id2cls_id, tokenizer
        )
        print(tokenized_inputs)
        ```
    """

    # Sample examples with 70% train and 30% test with equal distribution of labels
    num_positive = len(dataset.filter(lambda example: example["labels"] == 1))
    num_negative = len(dataset.filter(lambda example: example["labels"] == 0))
    train_negatives = (
        dataset.filter(lambda example: example["labels"] == 0)
        .shuffle(seed=seed)
        .select(range(int(num_negative * 0.7)))
    )
    train_positives = (
        dataset.filter(lambda example: example["labels"] == 1)
        .shuffle(seed=seed)
        .select(range(int(num_positive * 0.7)))
    )
    train = concatenate_datasets([train_negatives, train_positives])
    test_negatives = (
        dataset.filter(lambda example: example["labels"] == 0)
        .shuffle(seed=seed)
        .select(range(int(num_negative * 0.7), num_negative))
    )
    test_positives = (
        dataset.filter(lambda example: example["labels"] == 1)
        .shuffle(seed=seed)
        .select(range(int(num_positive * 0.7), num_positive))
    )
    test = concatenate_datasets([test_negatives, test_positives])
    return train, test


def split_dataset_by_type(dataset, train_percent=0.5):
    """
    Splits a dataset into training and testing sets based on paraphrase types.

    Args:
        dataset (dict): The dataset to split.
        train_percent (float, optional): The percentage of data to allocate for training.
        Defaults to 0.5.

    Returns:
        tuple: A tuple containing two lists: train and test. train contains the training examples,
        and test contains the testing examples.

    Example:
        ```python
        dataset = {
            "paraphrase_types": [["type1"], ["type2"]],
            "examples": [
                {"paraphrase_types": ["type1"]},
                {"paraphrase_types": ["type2"]},
                {"paraphrase_types": ["type1", "type2"]},
            ]
        }
        train, test = split_dataset_by_type(dataset, train_percent=0.7)
        print(train)
        print(test)
        ```
    """

    train = []
    test = []
    counts = {}
    for paraphrase_type in dataset["paraphrase_types"]:
        for par_type in paraphrase_type:
            if par_type not in counts:
                counts[par_type] = 1
            else:
                counts[par_type] += 1
    internal_counts = {key: 0 for key in counts}

    for example in dataset:
        types = example["paraphrase_types"]
        if len(types) > 0:
            # Get the type which has the lowest internal count
            par_type = min(types, key=lambda x: internal_counts[x])
            if internal_counts[par_type] < counts[par_type] * (1 - train_percent):
                test.append(example)
            else:
                train.append(example)
            for par_types in types:
                internal_counts[par_types] += 1

    return train, test


def parse_args():
    """
    Parses command line arguments.

    Returns:
        argparse.Namespace: An object containing the parsed arguments.

    Example:
        ```python
        args = parse_args()
        print(args.model_name)
        print(args.dataset_name)
        print(args.task_name)
        ```
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-large-uncased",
        help="Name of the model to use",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="jpwahle/etpc",
        help="Name of the dataset to use",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="paraphrase-detection",
        help="Name of the task to use",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Hardware to use, either of 'cpu', 'cuda', or 'mps'",
    )
    return parser.parse_args()


def tokenize_fn(examples, sentence1_key, sentence2_key, tokenizer):
    """
    Tokenizes input examples using a tokenizer.

    Args:
        examples (dict): The input examples.
        sentence1_key (str): The key for the first sentence in the examples.
        sentence2_key (str, optional): The key for the second sentence in the examples.
        Defaults to None.
        tokenizer (Tokenizer): The tokenizer to use for tokenization.

    Returns:
        dict: The tokenized inputs.

    Example:
        ```python
        examples = {
            "sentence1": "This is sentence 1",
            "sentence2": "This is sentence 2",
            "etpc_label": 1
        }
        sentence1_key = "sentence1"
        sentence2_key = "sentence2"
        tokenizer = Tokenizer()
        tokenized_inputs = tokenize_fn(examples, sentence1_key, sentence2_key, tokenizer)
        print(tokenized_inputs)
        ```
    """

    tokenized_inputs = tokenizer(
        *(
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        ),
        padding="max_length",
        truncation=True,
    )
    tokenized_inputs["labels"] = examples["etpc_label"]

    return tokenized_inputs


def main():
    args = parse_args()

    # Load the dataset
    dataset = load_dataset(args.dataset_name)

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, add_prefix_space=True)

    if "etpc" in args.dataset_name:
        # ETPC dataset keys
        sentence1_key = "sentence1"
        sentence2_key = "sentence2"
        dataset = dataset["train"]
    elif "qqp" in args.dataset_name:
        # QQP dataset keys
        sentence1_key = "question1"
        sentence2_key = "question2"

    # For the paraphrase_type_detection task, we need to create a data collator
    data_collator = None
    compute_metrics = None

    # Split the dataset into train and test
    if args.task_name == "paraphrase-type-detection":
        # Get label maps
        (
            label2cls_id,
            cls_id2label,
            paraphrase_type2cls_id,
            paraphrase_id2cls_type,
            paraphrase_type_to_category,
            cls_id2paraphrase_type_id,
            paraphrase_type_id2cls_id,
        ) = create_label_maps(dataset)

        # Tokenize the dataset
        dataset_tokenized = dataset.map(
            tokenize_and_align_labels,
            batched=True,
            fn_kwargs={
                "sentence1_key": sentence1_key,
                "sentence2_key": sentence2_key,
                "tokenizer": tokenizer,
                "paraphrase_type_id2cls_id": paraphrase_type_id2cls_id,
            },
        )

        # Train test split
        train, test = split_dataset_by_type(dataset_tokenized)

        # For Paraphrase Type Detection we need a data collator for token-level classifiction
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

        # Define metric
        metric = evaluate.load("seqeval")

        def compute_metrics_types(p):
            predictions, labels = p
            predictions = np.argmax(predictions, axis=2)

            # Remove ignored index (special tokens)
            true_predictions = [
                [
                    cls_id2label[p]
                    for (p, lab) in zip(prediction, label)
                    if lab not in [-100, 0]
                ]
                for prediction, label in zip(predictions, labels)
            ]
            true_labels = [
                [
                    cls_id2label[lab]
                    for (p, lab) in zip(prediction, label)
                    if lab not in [-100, 0]
                ]
                for prediction, label in zip(predictions, labels)
            ]

            results = metric.compute(
                predictions=true_predictions, references=true_labels
            )
            return results

        # Set metric
        compute_metrics = compute_metrics_types

        # Create a model that predicts the paraphrase_type of a sentence pair
        model = AutoModelForTokenClassification.from_pretrained(
            args.model_name,
            num_labels=27,
            id2label=cls_id2label,
            label2id=label2cls_id,
        ).to(args.device)

    elif args.task_name == "paraphrase-detection":
        # Tokenize the dataset
        dataset_tokenized = dataset.map(
            tokenize_fn,
            batched=True,
            fn_kwargs={
                "sentence1_key": sentence1_key,
                "sentence2_key": sentence2_key,
                "tokenizer": tokenizer,
            },
        )

        # Train test split
        train, test = split_dataset_binary(dataset_tokenized)

        # Create model
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=2,
            id2label={0: "no_paraphrase", 1: "paraphrase"},
            label2id={"no_paraphrase": 0, "paraphrase": 1},
        ).to(args.device)

        # Define metric
        metric = evaluate.load("f1")

        def compute_metrics_binary(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(
                predictions=predictions, references=labels, average="micro"
            )

        # Set metric
        compute_metrics = compute_metrics_binary

    else:
        raise NotImplementedError(f"Task {args.task_name} not implemented.")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"./out/cls-models/{args.model_name}-{args.dataset_name}-{args.task_name}",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to=None,
        use_mps_device=args.device == "mps",
    )

    # Traininer object
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
    trainer.train()

    # Evaluation
    results = trainer.evaluate()
    print("#" * 20)
    print(args.model_name)
    print(args.task_name)
    print(results)
    print("#" * 20)

    # Map results to lists in case of scalar values
    if args.task_name == "paraphrase-detection":
        results = pd.DataFrame(results, index=[0])
    else:
        results = pd.DataFrame(results)

    # Store results
    results.to_csv(
        f"{args.model_name}-{args.dataset_name}-paraphrase-{args.task_name}-results.csv",
    )


if __name__ == "__main__":
    main()
