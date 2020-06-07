from dataclasses import dataclass, field

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, HfArgumentParser
from utils_ner import get_labels
from seqeval.metrics.sequence_labeling import get_entities


@dataclass
class PredArguments:
    model_path: str = field(
        default="./model",
        metadata={"help": "Model saved by run_ner.py"}
    )
    labels: str = field(
        default="./data/labels.txt",
        metadata={"help": "Path to a file containing all labels."}
    )
    input: str = field(
        default="在福特的帮助下，阿瑟·登特在地球被毁灭前的最后一刻搭上了一艘路过地球的外星人的太空船，远离这个即将毁灭的伤心地，开始了一段充满惊奇的星河探险",
        metadata={"help": "Input data"}
    )


if __name__ == "__main__":
    parser = HfArgumentParser(PredArguments)
    args = parser.parse_args_into_dataclasses()[0]

    model = AutoModelForTokenClassification.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    labels = get_labels(args.labels)

    inputs = tokenizer.encode(args.input, return_tensors="pt")
    outputs = model(inputs)[0]
    prediction = torch.argmax(outputs, dim=2)[0]

    seq = [labels[it] for it in prediction.tolist()]
    # Bit of a hack to get the tokens with the special tokens
    tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(args.input)))
    entities = get_entities(seq)
    print(f"Input: {args.input}")
    for entity in entities:
        print(f"{entity}: {''.join(tokens[entity[1]:entity[2] + 1])}")
