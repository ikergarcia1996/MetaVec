import os
import time
import shutil
import argparse
from typing import List, Set
from tqdm import tqdm
from MetaVec.utils import get_vocab
from shlex import quote
from MetaVec.embedding import Embedding
from MetaVec.matrix_math import get_average_length


def generate_dictionary(embedding1: str, embedding2: str, output_path: str):
    vocab_intersection: Set[str] = get_vocab(embedding1).intersection(
        get_vocab(embedding2)
    )
    with open(output_path, "w+", encoding="utf8") as output_file:
        print(
            "\n".join([f"{word} {word}" for word in vocab_intersection]),
            file=output_file,
        )


def run_vecmap(
    embeddings_path: List[str],
    rotate_to: str,
    output_dir: str,
    normalize_method: str = "l2",
    clean_files: bool = True,
    use_cuda: bool = False,
):

    assert normalize_method in ["none", "vecmap_default", "scale", "l2"], (
        f"Normalization method {normalize_method} not supported"
        f"chose from [none, vecmap_default, scale]"
    )

    print("---> Running Vecmap <---")

    tmp_dir: str = os.path.join(output_dir, f"vecmap_tmp_{time.time()}")
    print(f"[Step1] Creating tmp directory in {tmp_dir}")
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    print(f"[Step2] Embedding normalization")

    if normalize_method == "vecmap_default":
        print("Normalization method: Vecmap Default (L2 norm and mean center)")
        for embedding_path in tqdm(
            embeddings_path + [rotate_to]
            if rotate_to not in embeddings_path
            else embeddings_path,
            desc="Normalizing embedding",
        ):

            command: str = (
                f"python3 vecmap/normalize_embeddings.py unit center "
                f"-i {quote(embedding_path)} "
                f"-o {quote(os.path.join(tmp_dir, os.path.basename(embedding_path)))} "
            )

            print(command)

            os.system(command)

        command: str = (
            f"python3 vecmap/normalize_embeddings.py unit center "
            f"-i {quote(rotate_to)} "
            f"-o {quote(os.path.join(tmp_dir, os.path.basename(rotate_to)))}"
        )
        print(command)

        os.system(command)

    elif normalize_method == "scale":
        print("Normalization method: Translate + Scale")

        for embedding_path in tqdm(
            embeddings_path + [rotate_to]
            if rotate_to not in embeddings_path
            else embeddings_path,
            desc="Embedding Normalization",
        ):

            embedding: Embedding = Embedding.from_file(embedding_path)

            embedding.mean_center()

            avg_scale: float = get_average_length(embedding.vectors)

            embedding.scale_vectors(scale_factor=1.0 / avg_scale)

            embedding.save(path=os.path.join(tmp_dir, os.path.basename(embedding_path)))

    elif normalize_method == "l2":
        print("Normalization method: L2 + Mean Center + L2")
        for embedding_path in tqdm(
            embeddings_path + [rotate_to]
            if rotate_to not in embeddings_path
            else embeddings_path,
            desc="Embedding Normalization",
        ):

            embedding: Embedding = Embedding.from_file(embedding_path)
            embedding.normalize_length()
            embedding.mean_center()
            embedding.normalize_length()
            # embedding.scale_vectors(scale_factor=100.0)
            embedding.save(path=os.path.join(tmp_dir, os.path.basename(embedding_path)))

    print(f"[Step3] VecMap")

    for embedding_path in tqdm(embeddings_path, desc="Mapping embeddings:"):
        source: str = (
            embedding_path
            if normalize_method == "none"
            else os.path.join(tmp_dir, os.path.basename(embedding_path))
        )
        target: str = (
            rotate_to
            if normalize_method == "none"
            else os.path.join(tmp_dir, os.path.basename(rotate_to))
        )
        source_output: str = os.path.join(output_dir, os.path.basename(embedding_path))
        target_output: str = os.path.join(tmp_dir, "target.vec")
        dictionary_path: str = os.path.join(
            tmp_dir,
            f"dictionary_{os.path.basename(embedding_path)}_{os.path.basename(rotate_to)}.txt",
        )

        generate_dictionary(
            embedding1=source, embedding2=target, output_path=dictionary_path
        )

        command: str = (
            f"python3 MetaVec/vecmap/map_embeddings.py "
            f"--orthogonal {quote(source)} {quote(target)} {quote(source_output)} {quote(target_output)} "
            f"-d {quote(dictionary_path)} --batch_size 1000"
        )

        if use_cuda:
            command += " --cuda"

        print(command)

        os.system(command)

    print("Cleaning...")

    if normalize_method != "none":
        os.rename(
            os.path.join(tmp_dir, os.path.basename(rotate_to)),
            os.path.join(output_dir, os.path.basename(rotate_to)),
        )
    else:
        os.rename(
            os.path.join(tmp_dir, os.path.basename("target.vec")),
            os.path.join(output_dir, os.path.basename(rotate_to)),
        )

    if clean_files:
        shutil.rmtree(tmp_dir)

    print(f"Done!!! Embeddings exported to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source_embeddings",
        nargs="+",
        type=str,
        required=True,
        help="Path to the source embeddings that will be mapped to the target embeddings",
    )

    parser.add_argument(
        "--target_embedding",
        type=str,
        required=True,
        help="Path to the target embedding to which the source embeddings will be mapped",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory where the resulting embeddings will be stored",
    )

    parser.add_argument(
        "--normalize_method",
        type=str,
        choices=["vecmap_default", "none", "scale", "l2"],
        default="l2",
        help="Embedding length normalization method.\n"
        "vecmap_default: Normalize length and mean center\n"
        "scale: Scale embeddings (embedding * scale_factor) to match the embedding with max average vector length\n"
        "none: Do not normalize embeddings\n"
        "l2: Length normalization. Mean centering. Length normalization",
    )

    parser.add_argument(
        "--do_not_clean_files",
        action="store_true",
        help="Do not remove the normalized embeddings and "
        "dictionaries from the temporal folder in the output directory",
    )

    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Use cuda",
    )

    args = parser.parse_args()

    run_vecmap(
        embeddings_path=args.source_embeddings,
        rotate_to=args.target_embedding,
        output_dir=args.output_dir,
        normalize_method=args.normalize_method,
        clean_files=not args.do_not_clean_files,
        use_cuda=args.cuda,
    )
