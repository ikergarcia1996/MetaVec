import argparse
from typing import List, Optional

import numpy as np
from tqdm import tqdm

from MetaVec.embedding import Embedding
from MetaVec.matrix_math import matrix_add, length_normalize
import os


def get_dims(embedding_paths: List[str]) -> (int, int):
    num_words: List[int] = []
    dimensions: List[int] = []

    for embedding_path in embedding_paths:
        with open(embedding_path, "r", encoding="utf8") as emb_file:
            header: List[str] = emb_file.readline().rstrip().strip().split()
            if len(header) != 2:
                raise ValueError(
                    f"{embedding_path} not in the word2vec format."
                    f" First line should contain a header with the number of words"
                    f" in the embeddings and the dimensions of the vectors."
                )
            else:
                num_words.append(int(header[0]))
                dimensions.append(int(header[1]))

    if len(set(num_words)) != 1:
        raise ValueError(
            f"All embeddings must have the same number of words "
            f"{[f'Embedding: {embedding_paths[i]} has {w} words' for i,w in enumerate(num_words)]}"
        )

    if len(set(dimensions)) != 1:
        raise ValueError(
            f"All embeddings must have the same number of dimensions "
            f"{[f'Embedding: {embedding_paths[i]} has {w} dimensions' for i, w in enumerate(dimensions)]}"
        )

    return num_words[0], dimensions[0]


def average_embeddings(
    embedding_paths: List[str],
    normalize_before: bool = True,
    normalize_after: bool = True,
    fp: int = 32,
    output_path: str = None,
    return_embedding: bool = False,
    batch_size: int = 2000,
    scale: bool = True,
) -> Optional[Embedding]:

    print(f"--> Embedding average <--")
    print(
        f"-> normalize_before: {normalize_before}. normalize_after: {normalize_after}. fp{fp}."
    )
    assert (
        output_path or return_embedding
    ), "No output method specified. Provide a path to print the embedding or set the flag return_embedding"

    num_words, dimensions = get_dims(embedding_paths=embedding_paths)

    print(
        f"The resulting meta-embedding will have {num_words} words and {dimensions} dimensions"
    )

    if output_path and not os.path.exists(os.path.dirname(output_path)):
        print(
            f"output dir not found, we will create it: {os.path.dirname(output_path)}"
        )
        os.makedirs(os.path.dirname(output_path))

    vocabulary: List[str] = []
    vectors: np.ndarray = np.asarray([])

    for embedding_path in tqdm(embedding_paths, desc="Embedding average"):
        embedding: Embedding = Embedding.from_file(
            path=embedding_path,
            fp=fp,
        )

        if normalize_before:
            embedding.normalize_length()

        if not vocabulary:
            vocabulary = embedding.words
            vectors = embedding.words2matrix(vocabulary)

        else:
            vectors = matrix_add(
                vectors,
                embedding.words2matrix(vocabulary),
                show_progress=True,
                batch_size=batch_size,
            )

    if normalize_after:
        vectors = length_normalize(vectors, inplace=True)
    else:
        vectors /= len(embedding_paths)

    if scale:
        vectors = vectors * 100.0

    if output_path:

        print(f"--> Saving embedding to {output_path} <--")
        with open(output_path, "w+", encoding="utf-8") as output_file:
            print(f"{num_words} {dimensions}", file=output_file)
            for i in tqdm(range(len(vocabulary)), desc="Saving embedding"):
                print(
                    vocabulary[i] + " " + " ".join(["%.6g" % x for x in vectors[i]]),
                    file=output_file,
                )

        print(f"Embedding exported to : {output_path}")

    if return_embedding:
        return Embedding(vocabulary=vocabulary, vectors=vectors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--embeddings",
        nargs="+",
        type=str,
        required=True,
        help="Path to the embeddings that we will use to generate a meta-embeddings",
    )

    parser.add_argument(
        "--fp",
        type=int,
        choices=[16, 32, 64],
        default=32,
        help="FP precision to use for the computations",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=2000,
        help="Batch size to use in matrix operations",
    )

    parser.add_argument(
        "--do_not_normalize_before",
        action="store_false",
        help="Length normalize embeddings before averaging (Length normalization is highly recommended)",
    )

    parser.add_argument(
        "--do_not_normalize_after",
        action="store_false",
        help="Length normalize embeddings after averaging",
    )

    parser.add_argument(
        "--do_not_scale",
        action="store_false",
        help="Multiply final embeddings * 100 for extrinsic eval",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output path",
    )

    args = parser.parse_args()

    average_embeddings(
        embedding_paths=args.embeddings,
        normalize_before=args.do_not_normalize_before,
        normalize_after=args.do_not_normalize_after,
        fp=args.fp,
        output_path=args.output_path,
        return_embedding=False,
        batch_size=args.batch_size,
        scale=args.do_not_scale,
    )
