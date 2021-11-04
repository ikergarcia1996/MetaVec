import sys

sys.path.insert(0, "../MetaVec")
from embedding import Embedding
import subprocess
import argparse


def blocks(files, size=65536):
    while True:
        b = files.read(size)
        if not b:
            break
        yield b


def count_lines(input_path: str) -> int:
    with open(input_path, "r", encoding="utf-8") as f:
        return sum(bl.count("\n") for bl in blocks(f))


def run_bash_command(command: str) -> None:
    subprocess.run(["bash", "-c", command])


def glove2w2v(
    embedding_path: str, output_path: str = None, normalize_dimensionwise: bool = False
):

    num_lines: int = count_lines(embedding_path)
    dims: int
    with open(embedding_path, "r", encoding="utf8") as file:
        next(file)
        dims = len(file.readline().rstrip().split()) - 1

    command: str
    if output_path:
        command = f"sed '1s/^/{num_lines} {dims}\\n/' {embedding_path} > {output_path}"
    else:
        command = f"sed -i '1s/^/{num_lines} {dims}\\n/' {embedding_path}"

    print(command)
    run_bash_command(command)

    if normalize_dimensionwise:
        emb = Embedding.from_file(
            path=embedding_path if not output_path else output_path
        )
        emb.normalize_length_dimensionwise()
        emb.save(path=embedding_path if not output_path else output_path)


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

    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize the embedding dimensionwise as recommended by the glove authors",
    )

    args = parser.parse_args()

    glove2w2v(
        embedding_path=args.embedding_path,
        output_path=args.output_path,
        normalize_dimensionwise=args.normalize,
    )
