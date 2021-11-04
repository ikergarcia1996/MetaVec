from typing import List
import argparse


def clean(embedding_path: str, output_path: str = None, block_size: int = 665536):

    with open(embedding_path, "r", encoding="utf8", errors="ignore") as input_file:
        with open(output_path, "w+", encoding="utf8") as output_file:
            lines: List[str] = input_file.readlines(block_size)
            while lines:
                print(" ".join(lines), file=output_file)
                lines = input_file.readlines(block_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--embedding_path",
        type=str,
        required=True,
        help="Embedding in the glove format",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output path, if not provided we will overwrite the input embedding path",
    )

    args = parser.parse_args()

    clean(
        embedding_path=args.embedding_path,
        output_path=args.output_path,
    )
