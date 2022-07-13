import json
import os

from datasets import load_dataset
from tqdm import tqdm

qqp = load_dataset("glue", "qqp")

# Shuffle the dataset entries
train_data = qqp["train"]
test_data = qqp["test"]


def write_to_jsonl(
    data,
    filename,
    generation_message,
    detection_message,
    generation_examples=None,
    detection_exampels=None,
    few_shot=False,
    chain_of_thought=False,
):
    with open(filename, "w", encoding="utf-8") as file:
        if chain_of_thought:
            detection_message += (
                " Think step by step. Then finally answer with the above."
            )
            generation_message += (
                " Think step by step. Then finally answer with the above."
            )
        for instance in tqdm(data):
            # Construct detection entry
            if few_shot:
                detection_message += (
                    "Positive Examples:\n"
                    + "\n\n".join(
                        [
                            positive["input"] + "\n" + positive["output"]
                            for positive in detection_examples["positive_examples"]
                        ]
                    )
                    + "\n\nNegative Examples:\n"
                    + "\n\n".join(
                        [
                            negative["input"] + "\n" + negative["output"]
                            for negative in detection_examples["negative_examples"]
                        ]
                    )
                )

                generation_message += (
                    "Positive Examples:\n"
                    + "\n\n".join(
                        [
                            positive["input"] + "\n" + positive["output"]
                            for positive in generation_examples["positive_examples"]
                        ]
                    )
                    + "\n\nNegative Examples:\n"
                    + "\n\n".join(
                        [
                            negative["input"] + "\n" + negative["output"]
                            for negative in generation_examples["negative_examples"]
                        ]
                    )
                )

            detection_entry = {
                "messages": [
                    {
                        "role": "user",
                        "content": detection_message
                        + f"\nQuestion1: {instance['question1']}, Question2: {instance['question2']}\n",
                    },
                    {
                        "role": "assistant",
                        "content": "Yes" if instance["label"] == 1 else "No",
                    },
                ]
            }

            # Construct generation entry
            generation_entry = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"{generation_message}"
                        + f"\n{instance['question1']}\n",
                    },
                    {"role": "assistant", "content": instance["question2"]},
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

    detection_message = 'Here are two questions (Question1 and Question2). If these questions have the same meaning and same answer, answer "Yes", otherwise "No".'
    generation_message = "In this task you're given a question and you have to paraphrase the question to create the output question while retaining the meaning of the original question."
    detection_examples = {
        "positive_examples": [
            {
                "input": "Question1: How do I get into my Instagram if I forgot my email and my Facebook password?, Question2: I forgot my password and also my email password. how can I get back that account?",
                "output": "Yes",
                "explanation": 'These questions have the meaning and the same answer. So, the output should be "Yes".',
            },
            {
                "input": "Question1: Why don't Hong Kong residents emigrate from their cramped & stressful city, like to places such as Australia?, Question2: Why made Hong Kong so attractive to Britain as a colony given that it was the last of Britain's colonies and Britain does not profit from taxing Hong Kong?",
                "output": "No",
                "explanation": "The first question is about the emigration of Hong Kong residents and the second question is about the attraction of Hong Kong. So, they don't have the same meaning.",
            },
            {
                "input": "Question1: Where had King Pandu gone during Vanvas?, Question2: Why mercury has lower vapour pressure?",
                "output": "No",
                "explanation": 'In the first question, King Pandu is discussed while the second question deals with mercury\'s vapour pressure. So, the output should be "No".',
            },
        ],
        "negative_examples": [
            {
                "input": "Question1: Why are there so many accidents on I-880?, Question2: Were there accidents in outer space?",
                "output": "Yes",
                "explanation": 'Question1 asks about the cause of the accidents, while question2 inquires about their existence. So, they are different and the correct output should be "No".',
            },
            {
                "input": "Question1: How do you determine the number of neutrons of an element or its ion?, Question2: How do you find the number of neutrons in an element? What are some examples?",
                "output": "They are the same.",
                "explanation": 'Note that you need to answer with "Yes" or "No" and other answers are not acceptable.',
            },
        ],
    }
    generation_examples = {
        "positive_examples": [
            {
                "input": "What can one do after MBBS?",
                "output": "What do i do after my MBBS ?",
                "explanation": "In this example both the question ask the same thing about what to do after MBBS hence second question is the correct output ",
            },
            {
                "input": "Which is the best book to study TENSOR for general relativity from basic?",
                "output": "Which is the best book for tensor calculus?",
                "explanation": "In this example first question is asking for a good book for learning tensor hence the second question will be a valid output.",
            },
            {
                "input": "What are the coolest Android hacks and tricks you know?",
                "output": "What are some cool hacks for Android phones?",
                "explanation": "In this example question 1 is asking about some of the coolest Android hacks and tricks and the same is being done by the question 2. Hence, second question is a valid output.",
            },
            {
                "input": "Which are the best motivational videos?",
                "output": "What are some of the best motivational clips?",
                "explanation": "In this example second question is a valid example as both the questions are asking for the best motivational videos.",
            },
        ],
        "negative_examples": [
            {
                "input": "Do you need a passport to go to Jamaica from the United States?",
                "output": "How can I move to Jamaica?",
                "explanation": "In this example even though both questions have the same overall theme of moving to Jamaica the first question is only asking if Passport will be required for travelling to Jamaica from United States where the second quesition is just asking how do we move to Jamaica in general.Hence, second question is not a valid output.",
            },
            {
                "input": "How is the life of a math student? Could you describe your own experiences?",
                "output": "Which level of prepration is enough for the exam jlpt5?",
                "explanation": "In this example first question is asking about life of a math student wheras the second question is asking about preparation for jlpt5 exam. Since both do not mean the same thing this will not be a valid output.",
            },
        ],
    }

    # Write to JSONL files in the 'out' directory
    write_to_jsonl(
        train_data,
        "out/detection_train.jsonl",
        detection_message=detection_message,
        generation_message=generation_message,
        detection_exampels=detection_examples,
        generation_examples=generation_examples,
    )
    write_to_jsonl(
        test_data,
        "out/detection_test.jsonl",
        detection_message=detection_message,
        generation_message=generation_message,
        detection_exampels=detection_examples,
        generation_examples=generation_examples,
    )
    write_to_jsonl(
        train_data,
        "out/generation_train.jsonl",
        detection_message=detection_message,
        generation_message=generation_message,
        detection_exampels=detection_examples,
        generation_examples=generation_examples,
    )
    write_to_jsonl(
        test_data,
        "out/generation_test.jsonl",
        detection_message=detection_message,
        generation_message=generation_message,
        detection_exampels=detection_examples,
        generation_examples=generation_examples,
    )

    print("JSONL files created in 'out' directory successfully!")
