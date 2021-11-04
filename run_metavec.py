import os
from MetaVec.run_vecmap import run_vecmap
from MetaVec.oov_generator import oov_generator
from MetaVec.average_embeddings import average_embeddings
from typing import List
import tempfile
import argparse


def generate_metaembedding(
    embeddings_path: List[str], rotate_to: str, output_path: str
):

    with tempfile.TemporaryDirectory() as tmpdirname:
        vecmap_dir = os.path.join(tmpdirname, "alignment_step")
        os.makedirs(vecmap_dir)

        run_vecmap(
            embeddings_path=embeddings_path,
            rotate_to=rotate_to,
            output_dir=vecmap_dir,
            normalize_method="l2",
        )

        aligned_embeddings_path: List[str] = [
            os.path.join(vecmap_dir, os.path.basename(emb_path))
            for emb_path in embeddings_path
        ]
        oov_dir = os.path.join(tmpdirname, "oov_step")
        os.makedirs(oov_dir)

        oov_generator(
            embeddings_paths=aligned_embeddings_path,
            save_dir=oov_dir,
        )

        harmonized_embeddings_path: List[str] = [
            os.path.join(oov_dir, os.path.basename(emb_path))
            for emb_path in embeddings_path
        ]

        average_embeddings(
            embedding_paths=harmonized_embeddings_path,
            output_path=output_path,
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
        "--rotate_to",
        type=str,
        required=True,
        help="Path to the target embedding to which the source embeddings will be mapped",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the generated meta embedding",
    )

    args = parser.parse_args()

    generate_metaembedding(
        embeddings_path=args.embeddings,
        rotate_to=args.rotate_to,
        output_path=args.output_path,
    )
