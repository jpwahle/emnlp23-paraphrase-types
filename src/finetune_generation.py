import argparse

import torch
from datasets import load_dataset
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge import Rouge
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    PegasusForConditionalGeneration,
    Trainer,
    TrainingArguments,
)


def encode_data(original, target, tokenizer):
    """
    Encodes the original and target data using a tokenizer.

    Args:
        original (str): The original data.
        target (str): The target data.
        tokenizer (Tokenizer): The tokenizer to use.

    Returns:
        dict: The encoded inputs and labels.

    Example:
        ```python
        original = "This is the original data."
        target = "This is the target data."
        tokenizer = Tokenizer()
        encoded_data = encode_data(original, target, tokenizer)
        print(encoded_data)
        ```
    """
    inputs = tokenizer(
        original,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    targets = tokenizer(
        target,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": targets["input_ids"],
    }


class ParaphraseDataset(torch.utils.data.Dataset):
    """A dataset class for paraphrase generation.

    Args:
        data (list): The dataset.
        tokenizer (Tokenizer): The tokenizer to use.

    Example:
        ```python
        dataset = ParaphraseDataset(data, tokenizer)
        ```
    """

    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        sentence1 = row["question1"]
        sentence2 = row["question2"]

        # Tokenize sentence1 for input_ids
        input_data = self.tokenizer(
            sentence1,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        )

        # Tokenize sentence2 for labels
        label_data = self.tokenizer(
            sentence2,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        )

        return {
            "input_ids": input_data["input_ids"].squeeze(0),
            "attention_mask": input_data["attention_mask"].squeeze(0),
            "labels": label_data["input_ids"].squeeze(0),
        }


class ParaphraseTypeDataset(torch.utils.data.Dataset):
    """A dataset class for paraphrase type generation.

    Args:
        data (list): The dataset.
        tokenizer (Tokenizer): The tokenizer to use.

    Example:
        ```python
        dataset = ParaphraseTypeDataset(data, tokenizer)
        ```
    """

    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        sentence1 = self.add_paraphrase_type_tags(
            row["sentence1_tokenized"],
            row["sentence1_segment_location_indices"],
            row["paraphrase_type_ids"],
        )
        sentence2 = row["sentence2"]

        # Tokenize sentence1 for input_ids
        input_data = self.tokenizer(
            sentence1,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        )

        # Tokenize sentence2 for labels
        label_data = self.tokenizer(
            sentence2,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        )

        return {
            "input_ids": input_data["input_ids"].squeeze(0),
            "attention_mask": input_data["attention_mask"].squeeze(0),
            "labels": label_data["input_ids"].squeeze(0),
        }

    def add_paraphrase_type_tags(self, sentence, segment_indices, type_ids):
        """Adds paraphrase type tags to specific indices in a sentence.

        Args:
            sentence (list): The input sentence as a list of tokens.
            segment_indices (list): The indices of the segments to tag.
            type_ids (list): The corresponding type IDs for each segment.

        Returns:
            str: The modified sentence with paraphrase type tags added.

        Example:
            ```python
            sentence = ["This", "is", "a", "sentence", "."]
            segment_indices = [[0, 3]]
            type_ids = [1]
            modified_sentence = add_paraphrase_type_tags(sentence, segment_indices, type_ids)
            print(modified_sentence)
            ```
        """
        for indices, type_id in zip(segment_indices, type_ids):
            for index in indices:
                sentence[index] = f"<type-{type_id}>{sentence[index]}"
        return " ".join(sentence)


def parse_args():
    """
    Parses the command line arguments and returns the parsed arguments.

    Returns:
        argparse.Namespace: The parsed command line arguments.

    Example:
        ```python
        args = parse_args()
        print(args.model_name)
        ```"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/bart-large")
    parser.add_argument(
        "--task_name",
        type=str,
        default="paraphrase-type-generation",
        help="Name of the task to use",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Hardware to use, either of 'cpu', 'cuda', or 'mps'",
    )

    return parser.parse_args()


def calculate_bleu(reference, candidate):
    """
    Calculates the BLEU score between a reference sentence and a candidate sentence.

    Args:
        reference (list): The reference sentences.
        candidate (str): The candidate sentence.

    Returns:
        float: The BLEU score."""

    smoothing = (
        SmoothingFunction().method1
    )  # Using SmoothingFunction's method1 for avoiding division by zero
    return sentence_bleu([reference], candidate, smoothing_function=smoothing)


def evaluate(predictions, targets):
    """
    Evaluates the predictions against the target references using BLEU and ROUGE scores.

    Args:
        predictions (list): The predicted sentences.
        targets (list): The target reference sentences.

    Returns:
        dict: A dictionary containing the BLEU score, ROUGE-1 score, ROUGE-2 score,
        and ROUGE-L score.

    Example:
        ```python
        predictions = ["This is a predicted sentence."]
        targets = ["This is a target sentence."]

        evaluation_results = evaluate(predictions, targets)
        print(evaluation_results)
        ```"""

    bleu_score = 0.0
    rouge_scores = {
        "rouge-1": {"f": 0.0},
        "rouge-2": {"f": 0.0},
        "rouge-l": {"f": 0.0},
    }

    if predictions and len(predictions) > 0:
        # BLEU Score
        for taget, prediction in zip(targets, predictions):
            b_sc = calculate_bleu(taget, prediction)
            bleu_score += b_sc
        bleu_score /= len(predictions)

        # ROUGE Scores
        rouge_calculator = Rouge()
        rouge_scores = rouge_calculator.get_scores(predictions, targets, avg=True)

    return {
        "bleu": bleu_score,
        "rouge-1": rouge_scores["rouge-1"]["f"],
        "rouge-2": rouge_scores["rouge-2"]["f"],
        "rouge-l": rouge_scores["rouge-l"]["f"],
    }


def eval_loop(data_loader, model, tokenizer):
    """
    Performs evaluation on a given data loader using a pre-trained model and tokenizer.

    Args:
        data_loader: The data loader object.
        model: The pre-trained model used for evaluation.
        tokenizer: The tokenizer object used for encoding and decoding.

    Returns:
        dict: A dictionary containing the evaluation results.

    Example:
        ```python
        data_loader = DataLoader(dataset, batch_size=32)
        model = PretrainedModel()
        tokenizer = Tokenizer()

        evaluation_results = eval_loop(data_loader, model, tokenizer)
        print(evaluation_results)
        ```"""
    model.eval()

    avg_scores = {"bleu": [], "rouge-1": [], "rouge-2": [], "rouge-l": []}

    with torch.no_grad():
        for batch in tqdm(data_loader):
            inputs = batch["input_ids"].to(model.device)
            attention_masks = batch["attention_mask"].to(model.device)
            outputs = model.generate(inputs, attention_mask=attention_masks)

            # Convert to text
            pred_texts = [
                tokenizer.decode(output, skip_special_tokens=True) for output in outputs
            ]
            target_texts = [
                tokenizer.decode(target, skip_special_tokens=True)
                for target in batch["labels"]
            ]
            scores = evaluate(pred_texts, target_texts)
            for key, value in scores.items():
                avg_scores[key].append(value)

    for key, value in avg_scores.items():
        avg_scores[key] = sum(value) / len(value)

    return avg_scores


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    model = (
        BartForConditionalGeneration.from_pretrained(args.model_name)
        if "bart" in args.model_name
        else PegasusForConditionalGeneration.from_pretrained(args.model_name)
    )

    if args.task_name == "paraphrase-type-generation":
        dataset = load_dataset("jpwahle/etpc").filter(
            lambda x: x["etpc_label"] == 1
        )  # Only positive paraphrases for training types
        # Split dataset['train'] into train and validation
        dataset = dataset["train"].train_test_split(test_size=0.2)
        train_dataset = ParaphraseTypeDataset(dataset["train"], tokenizer)
        eval_dataset = ParaphraseTypeDataset(dataset["test"], tokenizer)
    elif args.task_name == "paraphrase-generation":
        dataset = load_dataset("glue", "qqp")
        train_dataset = ParaphraseDataset(dataset["train"], tokenizer)
        eval_dataset = ParaphraseDataset(dataset["validation"], tokenizer)

    # Fine-tuning
    # training_args = TrainingArguments(
    #     per_device_train_batch_size=8,
    #     num_train_epochs=3,
    #     logging_dir="./logs",
    #     logging_steps=500,
    #     do_train=True,
    #     evaluation_strategy="epoch",
    #     save_steps=2000,
    #     save_total_limit=2,
    #     output_dir=f"./out/gen-models/{args.model_name}",
    #     use_mps_device=args.device == "mps",
    # )

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    # )
    # trainer.train()

    val_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=8)
    metrics = eval_loop(val_loader, model, tokenizer)

    # Evaluation
    print("#" * 20)
    print(args.model_name)
    print(args.task_name)
    print(metrics)
    print("#" * 20)


if __name__ == "__main__":
    main()
