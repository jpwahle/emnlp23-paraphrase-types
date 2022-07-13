import json
import os
import random

from datasets import load_dataset
from tqdm import tqdm

etpc = load_dataset("jpwahle/etpc")

# Shuffle the dataset entries
dataset_list = list(etpc["train"])
random.shuffle(dataset_list)

# Split the dataset into 70% training and 30% testing
train_length = int(0.7 * len(dataset_list))
train_data = dataset_list[:train_length]
test_data = dataset_list[train_length:]


def write_to_jsonl(data, filename):
    with open(filename, "w", encoding="utf-8") as file:
        for instance in tqdm(data):
            # Check if there are any paraphrase types in instance["paraphrase_types"]
            # otherwise skip
            if not instance["paraphrase_types"]:
                continue

            # Construct detection entry
            detection_entry = {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Given the following two sentences, which of the"
                            " paraphrase types are changed between them?"
                            f" Sentence 1: {instance['sentence1']} Sentence 2:"
                            f" {instance['sentence2']} Paraphrase Types:"
                            " Derivational Changes, Inflectional Changes,"
                            " Modal Verb Changes, Spelling changes, Change of"
                            " format, Same Polarity Substitution"
                            " (contextual), Same Polarity Substitution"
                            " (habitual), Same Polarity Substitution (named"
                            " ent.), Converse substitution, Opposite polarity"
                            " substitution (contextual), Opposite polarity"
                            " substitution (habitual), Synthetic/analytic"
                            " substitution, Coordination changes, Diathesis"
                            " alternation, Ellipsis, Negation switching,"
                            " Subordination and nesting changes,"
                            " Direct/indirect style alternations, Punctuation"
                            " changes, Syntax/discourse structure changes,"
                            " Entailment, Identity, Non-paraphrase,"
                            " Addition/Deletion, Change of order,"
                            " Semantic-based"
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": ", ".join(instance["paraphrase_types"]),
                    },
                ]
            }

            # Construct generation entry
            generation_entry = {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Given the following sentence, generate a"
                            " paraphrase with the following types. Sentence:"
                            f" {instance['sentence1']} Paraphrase Types:"
                            f" {', '.join(instance['paraphrase_types'])}"
                        ),
                    },
                    {"role": "assistant", "content": instance["sentence2"]},
                ]
            }

            # Write entries to the respective files
            if "detection" in filename:
                file.write(json.dumps(detection_entry) + "\n")
            else:
                file.write(json.dumps(generation_entry) + "\n")


if __name__ == "__main__":
    # Create output directory if it doesn't exist
    if not os.path.exists("out"):
        os.makedirs("out")

    # Write to JSONL files in the 'out' directory
    write_to_jsonl(train_data, "out/detection_train.jsonl")
    write_to_jsonl(test_data, "out/detection_test.jsonl")
    write_to_jsonl(train_data, "out/generation_train.jsonl")
    write_to_jsonl(test_data, "out/generation_test.jsonl")

    print("JSONL files created in 'out' directory successfully!")
