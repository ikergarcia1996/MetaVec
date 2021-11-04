from typing import List
import logging
import subprocess
import argparse


def run_bash_command(command: str) -> None:
    subprocess.run(["bash", "-c", command])


def remove_prefix(
    embedding_path: str,
    output_path: str,
    prefix_separator: str = "/",
    allowed_prefixes: List[str] = None,
    block_size: int = 665536,
):

    dimensions: int = 0
    printed_words: int = 0
    with open(embedding_path, "r", encoding="utf8") as input_file:
        with open(output_path, "w+", encoding="utf8") as output_file:
            line_no: int = 0
            try:
                header: str = input_file.readline()
                num_words, dimensions = header.split(" ")
                num_words, dimensions = int(num_words), int(dimensions)
            except ValueError as err:
                raise ValueError(
                    f"Error reading header. "
                    f"Expecting embedding in the word2vec format. "
                    f"Header expected: num_words dims. Header found: {header}.\n"
                    f"Error: {err}"
                )

            lines: List[str] = input_file.readlines(block_size)
            formatted_lines: List[str] = []
            while lines:
                for line in lines:

                    line_no += 1

                    l: List[str] = line.rstrip().split(" ")
                    word, vector = (
                        l[0].strip(),
                        l[1:],
                    )

                    if len(vector) != dimensions:
                        logging.warning(
                            f"Dimensions mismatch ({len(l)} in line {line_no}.. Skipping line. {l[:3]}..."
                        )
                        continue

                    if word == "":
                        logging.warning(
                            f"Error in line {line_no}. Skipping line. {l[:3]}..."
                        )
                        continue

                    prefix, word = word.split(prefix_separator, 1)

                    if allowed_prefixes is None or prefix in allowed_prefixes:
                        formatted_lines.append(f"{word} {' '.join(vector)}")

                printed_words += len(formatted_lines)
                if formatted_lines:
                    print("\n".join(formatted_lines), file=output_file)

                formatted_lines = []

                lines = input_file.readlines(block_size)

    command: str = f"sed -i '1s/^/{printed_words} {dimensions}\\n/' {output_path}"
    print(command)
    run_bash_command(command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--embedding_path",
        type=str,
        required=True,
        help="Embedding to format (word2vec format)",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path where the formatted embedding will be written",
    )

    parser.add_argument(
        "--prefix_separator",
        type=str,
        default="/",
        help="Path where the formatted embedding will be written",
    )

    parser.add_argument(
        "--allowed_prefixes",
        nargs="+",
        type=str,
        help="Allowed prefixes "
        "(words with other prefixes will be discarded, if allowed_prefixes=None we will allow all prefixes)",
    )

    args = parser.parse_args()

    remove_prefix(
        embedding_path=args.embedding_path,
        output_path=args.output_path,
        prefix_separator=args.prefix_separator,
        allowed_prefixes=args.allowed_prefixes,
    )
