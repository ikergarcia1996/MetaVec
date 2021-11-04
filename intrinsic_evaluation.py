import os
from typing import List
import subprocess
from shlex import quote
from tqdm import tqdm
import argparse


def eval_embedding(eval_script: str, embedding_path: str, output_path: str):
    command = (
        f"python3 {quote(eval_script)} "
        f"-f {quote(os.path.abspath(embedding_path))} "
        f"-p word2vec "
        f"-o {quote(os.path.abspath(output_path))}"
    )
    print(command)
    subprocess.run(["bash", "-c", command])


def eval_directory(embs_dir: str, eval_script: str, output_dir: str):

    emb_list: List[str] = [
        os.path.join(embs_dir, f)
        for f in os.listdir(embs_dir)
        if os.path.isfile(os.path.join(embs_dir, f))
    ]

    for emb in tqdm(emb_list, desc="Evaluating embeddings"):
        eval_embedding(
            eval_script=eval_script,
            embedding_path=emb,
            output_path=os.path.join(
                output_dir, f"{''.join(os.path.basename(emb).split('.')[:-1])}.txt"
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "-i",
        "--embedding_path",
        type=str,
        help="Embedding to evaluate",
    )

    group.add_argument(
        "-d",
        "--directory_path",
        type=str,
        help="Embedding to evaluate",
    )

    parser.add_argument(
        "-s",
        "--eval_script",
        type=str,
        default="word-embeddings-benchmarks/scripts/evaluate_on_all.py",
        help="Evaluate script (Path to: word-embeddings-benchmarks/scripts/evaluate_on_all.py)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="intrinsic_eval_results/",
        help="Directory where the evaluation results will be saved",
    )

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.embedding_path:
        eval_embedding(
            eval_script=args.eval_script,
            embedding_path=args.embedding_path,
            output_path=os.path.join(
                args.output_dir,
                f"{''.join(os.path.basename(args.embedding_path).split('.')[:-1])}.txt",
            ),
        )
    else:
        eval_directory(
            embs_dir=args.directory_path,
            eval_script=args.eval_script,
            output_dir=args.output_dir,
        )
