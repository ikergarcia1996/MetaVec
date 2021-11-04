from typing import Type, List, Set, Sized, Iterable
import numpy as np
import math


def batch(iterable: Sized, n: int = 1) -> Iterable:
    l = len(iterable)
    s = math.ceil(l / n)
    for ndx in range(0, l, s):
        yield iterable[ndx : min(ndx + s, l)]


def get_fp_type_numpy(fp: int) -> Type:
    if fp == 16:
        return np.float16
    elif fp == 32:
        return np.float32
    elif fp == 64:
        return np.float64
    else:
        raise ValueError(f"FP{fp} not supported. Supported values: 16, 32, 64.")


def get_vocab(embedding_path: str, block_size: int = 665536) -> Set[str]:
    vocab: List[str] = []
    with open(embedding_path, "r", encoding="utf8") as embedding:

        try:
            header: str = embedding.readline()
            num_words, dimensions = header.split(" ")
            num_words, dimensions = int(num_words), int(dimensions)
        except ValueError as err:
            raise ValueError(
                f"Error reading header. "
                f"Expecting embedding in the word2vec format. "
                f"Header expected: num_words dims. Header found: {header}.\n"
                f"Error: {err}"
            )

        lines: List[str] = embedding.readlines(block_size)
        while lines:
            for line in lines:
                l: List[str] = line.rstrip().split(" ")
                word, vector = (
                    l[0].strip(),
                    l[1:],
                )

                if len(vector) != dimensions:
                    continue

                if word == "":
                    continue

                vocab.append(word)

            lines = embedding.readlines(block_size)

    return set(vocab)
