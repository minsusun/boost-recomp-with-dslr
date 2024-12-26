import json
import builtins
from argparse import ArgumentParser

import pandas as pd
from tqdm import tqdm

import prompt_types


def warn(*args, **kargs):
    builtins.print("\033[33m[ * ]", *args, **kargs)
    builtins.print("\033[37m", end="")


def verbose(*args, **kargs):
    builtins.print("\033[90m[ + ]", *args, **kargs)
    builtins.print("\033[37m", end="")


def main():
    parser = ArgumentParser()
    parser.add_argument("-d", "--data", dest="data", type=str, required=True)
    parser.add_argument(
        "-p",
        "--prompt_type",
        dest="prompt_type",
        required=True,
        type=str,
        choices=["RECOMP", "DSLR"],
    )
    parser.add_argument(
        "-m",
        "--method",
        dest="method",
        choices=["RECOMP", "DSLR"],
        type=str,
        required=True,
    )
    parser.add_argument("-o", "--output", dest="output", type=str, required=True)
    parser.add_argument("-k", "--top_k", dest="top_k", default=5, type=int)
    parser.add_argument("-v", "--verbose", dest="verbose", action="count", default=0)

    args = parser.parse_args()
    if args.verbose:
        s = json.dumps(vars(args), indent=4)
        for line in s.split("\n"):
            verbose(line)

    base_prompt = getattr(prompt_types, args.prompt_type)
    if args.verbose:
        verbose("Selcted Prompt Type: {type}".format(type=args.prompt_type))

    data = pd.read_json(args.data)
    if args.verbose:
        verbose("Done reading {data}".format(data=args.data))

    target_score_col = [col for col in data.columns if "score" in col]
    assert (
        len(target_score_col) > 0
    ), "There is no score info, update the data with running the compressor"
    if len(target_score_col) > 1:
        warn("Number of scored colum is bigger than 2, we will first one as score")
    if args.verbose:
        verbose("Target Score Columns: {target}".format(target=target_score_col))

    output = []

    for _, row in tqdm(data.iterrows(), total=len(data)):
        assert len(row["sentence"]) == len(
            row[target_score_col[0]]
        ), "Number of sentences mismatch"

        question = row["query"]
        gold_answers = row["gold_answers"]

        sentence_score_list = [
            (idx, sentence, score)
            for idx, (sentence, score) in enumerate(
                zip(row["sentence"], row[target_score_col[0]])
            )
        ]
        sentence_score_list = sorted(
            sentence_score_list, key=lambda x: x[2], reverse=True
        )  # sort base on score - reranking

        doc_list = sentence_score_list[: args.top_k]
        if args.method == "DSLR":
            doc_list = sorted(
                doc_list, key=lambda x: x[0]
            )  # sort base on index - context reconstruction

        if args.verbose > 1:
            verbose("Document summary: ")
            for idx, sentence, score in doc_list:
                verbose(f"{{{idx:3d}}} {score:02.2f} {sentence[:50]}")

        doc = "\n".join([sentence for _, sentence, _ in doc_list])

        prompt = base_prompt.format(query=question, doc=doc)

        output.append((question, gold_answers, prompt))

    output = pd.DataFrame(output, columns=["question", "gold_answers", "prompt"])
    if args.verbose:
        verbose("Created Pandas Data Frame")
    output.to_csv(args.output, sep=",", index=False)
    if args.verbose:
        verbose("Saved {output}".format(output=args.output))


if __name__ == "__main__":
    main()
