from typing import Dict, Set, List
from MetaVec.embedding import Embedding
import MetaVec.matrix_math as matrix_math
import argparse
import os
from collections import Counter
from tqdm import tqdm
import numpy as np
from MetaVec.utils import batch


def get_union_vocab(
    list_of_vocabularies: List[List[str]], min_freq: int = 1
) -> Set[str]:
    if min_freq > 1:
        c: Counter[str, int] = Counter()
        for list_vocab in list_of_vocabularies:
            c.update(list_vocab)

        return set([word for word, freq in c.items() if freq > min_freq])
    else:
        return set.union(*[set(list_vocab) for list_vocab in list_of_vocabularies])


def save_meta_embedding(
    embedding: Embedding,
    new_words: List[str],
    new_vectors: np.ndarray,
    save_path: str,
):

    with open(save_path, "w+", encoding="utf-8") as output_file, tqdm(
        total=len(embedding) + len(new_words), desc=f"Saving embedding to {save_path}"
    ) as pbar:
        print(f"{len(embedding)+len(new_words)} {embedding.dims}", file=output_file)

        for words in batch(embedding.words, n=100):
            print(
                "\n".join(
                    [
                        word
                        + " "
                        + " ".join(["%.6g" % x for x in embedding.word2vector(word)])
                        for word in words
                    ]
                ),
                file=output_file,
            )

            pbar.update(len(words))

        for words, vectors in zip(batch(new_words, n=100), batch(new_vectors, n=100)):
            print(
                "\n".join(
                    [
                        word + " " + " ".join(["%.6g" % x for x in vector])
                        for word, vector in zip(words, vectors)
                    ]
                ),
                file=output_file,
            )
            pbar.update(len(words))

    print(f"Embedding exported to : {save_path}")


def oov_generator(
    embeddings_paths: List[str],
    save_dir: str,
    min_k_value: int = 2,
    max_k_value: int = 50,
    metric: str = "cosine",
    batch_size: int = 2000,
    min_freq_words: int = 1,
    normalize_method: str = "none",
    fp: int = 32,
    filter_words: bool = False,
    debug: bool = False,
):

    # PREPARE DATA

    assert normalize_method in [
        "l2",
        "none",
        "scale",
    ], f"Normalize method {normalize_method} not implemented. Available methods: [l2,none,scale]"

    assert (
        max_k_value >= min_k_value > 0
    ), f"max_k_value >= min_k_value > 0. min_k_value:{min_k_value}. max_k_value:{max_k_value}"

    # - Load embeddings

    embeddings = [
        Embedding.from_file(
            path,
            fp=fp,
            filter_words=filter_words,
        )
        for path in embeddings_paths
    ]

    # - Create vocabulary  and normalize embeddings

    dims = set([emb.dims for emb in embeddings])
    assert (
        len(dims) == 1
    ), f"All the embeddings should have the same number of dimensions. Dimensions found: {dims}"

    dims = dims.pop()

    normalize = normalize_method == "l2"

    union_vocabulary: Set[str] = get_union_vocab(
        [embedding.words for embedding in embeddings], min_freq=min_freq_words
    )

    v_sum = "\n".join(
        [
            f"{os.path.basename(ep)} has {len(emb)} words"
            for ep, emb in zip(embeddings_paths, embeddings)
        ]
    )
    print(
        f"\n--> Vocabulary summary <--\n"
        f"{v_sum}\n"
        f"- Union of vocabularies has {len(union_vocabulary)} words\n\n"
    )

    del v_sum

    if normalize_method == "l2":
        print("Normalization method: L2")
        for emb in embeddings:
            emb.normalize_length()
            emb.mean_center()
            emb.normalize_length()

    elif normalize_method == "scale":

        print("Normalization method: Translate + Scale")
        for embedding in tqdm(embeddings, desc="Embedding Normalization"):
            embedding.mean_center()
            avg_scale: float = matrix_math.get_average_length(embedding.vectors)
            embedding.scale_vectors(scale_factor=1.0 / avg_scale)

    # RUN THE OOV ALGORITHM

    print(f"--> Running OOV generation algorithm <--")
    print(
        f"Save dir: {save_dir}\n"
        f"No. Embeddings: {len(embeddings)}\n"
        f"Metric: {metric}\n"
        f"FP{fp}\n"
        f"Batch size. {batch_size}\n"
        f"min_k_value: {min_k_value}\n"
        f"max_k_value: {max_k_value}\n"
        f"Debug mode: {debug}\n"
        f"GPU available: {matrix_math.cupy_available}\n"
        f"GPU Free memory: {matrix_math.get_gpu_memory()} Mb \n\n"
    )

    for embedding_oov_id in range(len(embeddings)):
        print(f"--> Running OOV generation for embedding {embedding_oov_id} <--")
        embedding_oov = embeddings[embedding_oov_id]
        new_words: List[str] = list(
            union_vocabulary - embeddings[embedding_oov_id].set_words
        )
        print(
            f"We will generate {len(new_words)} words for embedding {embedding_oov_id}"
        )
        new_vectors: np.ndarray = np.zeros(
            (len(new_words), dims),
            dtype=embeddings[embedding_oov_id].embedding_numpy_type,
        )

        if len(new_words) == 0:
            return new_words, new_vectors

        new_words2index: Dict[str, int] = {w: i for i, w in enumerate(new_words)}

        new_vectors_distance: np.ndarray = np.full(
            (len(new_words)),
            fill_value=float("-inf") if metric == "cosine" else float("inf"),
            dtype=embeddings[embedding_oov_id].embedding_numpy_type,
        )

        for embedding_index_id in range(len(embeddings)):
            if embedding_index_id == embedding_oov:
                continue

            embedding_index = embeddings[embedding_index_id]
            words2generate: List[str] = list(
                embedding_index.set_words - embedding_oov.set_words
            )
            if len(words2generate) == 0:
                continue

            intersection_vocab: np.ndarray = np.asarray(
                list(embedding_index.set_words.intersection(embedding_oov.set_words))
            )

            if len(intersection_vocab) == 0:
                continue

            embedding_index_true_centroids: np.ndarray = embedding_index.words2matrix(
                words2generate
            )
            intersection_vocab_vectors: np.ndarray = embedding_index.words2matrix(
                intersection_vocab
            )
            if metric == "cosine":
                if normalize:
                    knn, knn_distances = matrix_math.knn_dot(
                        embedding_index_true_centroids,
                        intersection_vocab_vectors,
                        k=max_k_value,
                        return_distances=True,
                        ordered=True,
                        show_progress=True,
                        batch_size=batch_size,
                    )
                else:
                    knn, knn_distances = matrix_math.knn_cosine(
                        embedding_index_true_centroids,
                        intersection_vocab_vectors,
                        k=max_k_value,
                        return_distances=True,
                        ordered=True,
                        show_progress=True,
                        batch_size=batch_size,
                    )

            else:
                knn, knn_distances = matrix_math.knn_euclidean_distance(
                    embedding_index_true_centroids,
                    intersection_vocab_vectors,
                    k=max_k_value,
                    return_distances=True,
                    ordered=True,
                    show_progress=True,
                    batch_size=batch_size,
                )
                knn_distances = 1 / knn_distances

            for i in tqdm(range(len(words2generate)), desc=f"Generating OOV words: "):
                oov_word_id = new_words2index[words2generate[i]]
                candidates = matrix_math.rolling_average(
                    a=intersection_vocab_vectors[knn[i]],
                    weights=knn_distances[i] + 1,
                )[min_k_value - 1 : max_k_value]

                if metric == "cosine":

                    d: np.ndarray = matrix_math.cosine_similarity(
                        candidates, embedding_index_true_centroids[i][np.newaxis, :]
                    )
                    best_k = np.argmax(d)

                else:
                    if normalize:
                        candidates = matrix_math.length_normalize(candidates)

                    d: np.ndarray = matrix_math.euclidean_distance(
                        candidates, embedding_index_true_centroids[i][np.newaxis, :]
                    )

                    best_k = np.argmin(d)

                d = d[best_k][0]
                best_k = best_k + min_k_value

                if (metric == "cosine" and d > new_vectors_distance[oov_word_id]) or (
                    metric == "euclidean" and d < new_vectors_distance[oov_word_id]
                ):
                    new_vector = np.average(
                        embedding_oov.words2matrix(intersection_vocab[knn[i][:best_k]]),
                        weights=knn_distances[i][:best_k] + 1,
                        axis=0,
                    )
                    if normalize:
                        new_vector /= np.linalg.norm(new_vector)

                    prev_d = new_vectors_distance[oov_word_id]
                    new_vectors[oov_word_id] = new_vector
                    new_vectors_distance[oov_word_id] = d

                    if debug:
                        print(
                            f"Generating word {words2generate[i]} "
                            f"with nns {intersection_vocab[knn[i][:best_k]]}. "
                            # f"matrix: {embedding_oov.words2matrix(intersection_vocab[knn[i][:k]])}. "
                            # f"weights: {knn_distances[i][:k]}"
                            # f"v: {v[:10]}. "
                            f"Distance: {d}. "
                            f"Prev distance: "
                            f"{prev_d}"
                        )

        save_meta_embedding(
            embedding=embedding_oov,
            new_words=new_words,
            new_vectors=new_vectors,
            save_path=os.path.join(
                save_dir, os.path.basename(embeddings_paths[embedding_oov_id])
            ),
        )


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
        "--output_dir",
        type=str,
        required=True,
        help="Directory where we will store the resulting embedding",
    )

    parser.add_argument(
        "--metric",
        type=str,
        choices=["cosine", "euclidean"],
        default="cosine",
        help="use cosine similarity or euclidean distance to compute nearest nearest neighbours",
    )

    parser.add_argument(
        "--k_min_value",
        type=int,
        default=2,
        help="Min value to use in the k estimator grid search",
    )

    parser.add_argument(
        "--k_max_value",
        type=int,
        default=50,
        help="Max value to use in the k estimator grid search",
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
        "--min_freq_words",
        type=int,
        default=1,
        help="Min number of embeddings that need to have representation "
        "for a word to be included in the meta-embedding vocabulary",
    )

    parser.add_argument(
        "--filter_words",
        action="store_true",
        help="Remove uppercase words if a lowercase version exists and words that include symbols",
    )

    parser.add_argument(
        "--normalize_method",
        type=str,
        choices=["l2", "none", "scale", "scale_old"],
        default="none",
        help="Embedding length normalization method.\n"
        "l2: Normalize length using l2 norm\n"
        "scale: Scale embeddings (embedding * scale_factor) to match the embedding with max average vector length\n"
        "none: Do not normalize embeddings",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print the knn that we will use to generate each OOV word",
    )

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    oov_generator(
        embeddings_paths=args.embeddings,
        save_dir=args.output_dir,
        min_k_value=args.k_min_value,
        max_k_value=args.k_max_value,
        metric=args.metric,
        batch_size=args.batch_size,
        min_freq_words=args.min_freq_words,
        normalize_method=args.normalize_method,
        fp=args.fp,
        filter_words=args.filter_words,
        debug=args.debug,
    )
