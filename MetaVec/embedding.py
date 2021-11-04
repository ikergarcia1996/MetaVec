from typing import List, Type, Dict, Set, Union
import collections
import numpy as np
from MetaVec.utils import get_fp_type_numpy, get_vocab, batch
from tqdm import tqdm
from MetaVec.matrix_math import (
    length_normalize,
    length_normalize_dimensionwise,
    scale_matrix,
    mean_center,
)
import logging
from operator import itemgetter
import re
import os
import time


class Embedding:
    _embedding_numpy_type: Type
    _words: List[str]
    _set_words: Set[str]
    _vectors: np.ndarray
    _word2index: Dict[str, int]

    def __init__(
        self,
        vocabulary: List[str],
        vectors: Union[List[np.ndarray], np.ndarray],
        fp: int = 32,
    ):

        set_vocabulary: Set[str] = set(vocabulary)
        assert len(vocabulary) == len(set_vocabulary), (
            f"The provided vocabulary contains "
            f"{len(vocabulary)-len(set_vocabulary)} duplicated words! "
            f"The duplicated words are: "
            f"{[(item,count) for item, count in collections.Counter(vocabulary).items() if count > 1]}"
        )

        assert len(vocabulary) == len(vectors), (
            f"You have provided a different number of words and vectors!! "
            f"Number of words: {len(vocabulary)}. Number of vectors: {len(vectors)}"
        )

        vectors_len = set([len(x) for x in vectors])

        assert len(vectors_len) == 1, (
            f"All vectors must have the same number of dimensions. You provided a list of "
            f"vectors containing vectors with the following dimensions: {vectors_len}"
        )

        self._embedding_numpy_type: Type = get_fp_type_numpy(fp=fp)

        self._words: List[str] = vocabulary
        self._set_words: Set[str] = set_vocabulary
        self._vectors: np.ndarray = np.asarray(vectors, dtype=self.embedding_numpy_type)

        self._word2index: Dict[str, int] = {w: i for i, w in enumerate(self.words)}

    @classmethod
    def from_file(
        cls,
        path: str,
        fp: int = 32,
        vocabulary: Union[Set[str], str] = None,
        lowercase: bool = False,
        filter_words: bool = False,
    ):
        words, vectors = load_embedding(
            embedding_path=path,
            fp=fp,
            vocabulary=set(open(vocabulary, "r", encoding="utf8").readlines())
            if isinstance(vocabulary, str)
            else vocabulary,
            lowercase=lowercase,
            filter_words=filter_words,
        )

        return cls(vocabulary=words, vectors=vectors, fp=fp)

    def __len__(self) -> int:
        return len(self.words)

    @property
    def dims(self) -> int:
        return self.vectors.shape[1]

    @property
    def words(self) -> List[str]:
        return self._words

    @property
    def set_words(self) -> Set[str]:
        return self._set_words

    @property
    def vectors(self) -> np.ndarray:
        return self._vectors

    @property
    def embedding_numpy_type(self):
        return self._embedding_numpy_type

    def is_word_in_vocab(self, word: str):
        return word in self._set_words

    def normalize_length_dimensionwise(self):
        self._vectors = length_normalize_dimensionwise(self._vectors, inplace=True)

    def normalize_length(self):
        self._vectors = length_normalize(self._vectors, inplace=True)

    def mean_center(self):
        self._vectors = mean_center(self._vectors, inplace=True)

    def scale_vectors(self, scale_factor: float):
        self._vectors = scale_matrix(
            matrix=self.vectors, scale_factor=scale_factor, inplace=True
        )

    def word2vector(self, word: str) -> np.ndarray:
        return self._vectors[self._word2index[word]]

    def word2index(self, word: str) -> int:
        return self._word2index[word]

    def words2matrix(self, words) -> np.ndarray:
        if len(words) == 1:
            return self.word2vector(words[0])[
                np.newaxis,
            ]
        return self._vectors[np.asarray(itemgetter(*words)(self._word2index))]

    def words2matrix_backoff(
        self, words: List[str], backoff: np.ndarray, lower: bool = False
    ) -> np.ndarray:
        assert backoff.shape[0] == self.dims, (
            f"Backoff vector dimension {backoff.shape[0]} and embeddings "
            f"vectors dimensions {self.dims} must be the same. "
        )

        return np.asarray(
            [
                self.word2vector(word.lower() if lower else word)
                if (word.lower() if lower else word) in self._set_words
                else backoff
                for word in words
            ]
        )

    def add_word(self, word: str, vector: np.ndarray):
        self._words.append(word)
        self._vectors = np.concatenate((self._vectors, vector[np.newaxis, :]), axis=0)
        self._word2index[word] = self._vectors.shape[0] - 1
        self._set_words.update([word])

    def save(
        self,
        path: str,
        txt_format: bool = False,
    ):
        print(f"--> Saving embedding to {path} <--")
        with open(path, "w+", encoding="utf-8") as output_file, tqdm(
            total=len(self.words), desc="Saving embedding"
        ) as pbar:
            if not txt_format:
                print(f"{len(self)} {self.dims}", file=output_file)
            for words in batch(self.words, n=1000):
                print(
                    "\n".join(
                        [
                            word
                            + " "
                            + " ".join(["%.6g" % x for x in self.word2vector(word)])
                            for word in words
                        ],
                    ),
                    file=output_file,
                )
                pbar.update(len(words))

        print(f"Embedding exported to : {path}")


def valid_word(word: str, vocab: Set[str]) -> bool:
    # Remove uppercase words if a lowercase version exists
    # Remove words that include symbols

    if not word.islower():
        if word.lower() in vocab:
            return False

    if re.match("^[A-Za-z0-9]*$", word):
        return True

    return False


def format_lines(
    lowercase: bool,
    dimensions: int,
    vocabulary: Set[str],
    dtype: type,
    data: List[str],
):

    words: List[str] = []
    vectors: List[np.ndarray] = []

    for line in data:
        line = line.rstrip().split(" ")
        word, vector = (
            line[0].strip().lower() if lowercase else line[0].strip(),
            line[1:],
        )

        if len(vector) != dimensions:
            logging.warning(
                f"Dimensions mismatch ({len(vector)}.. Skipping line. {line[:3]}..."
            )
            continue

        if word == "":
            logging.warning(f"Error in line. Skipping line. {line[:3]}...")
            continue

        if vocabulary and word not in vocabulary:
            continue

        words.append(word)
        vectors.append(np.asarray(vector, dtype=dtype))

    return words, vectors


def load_embedding(
    embedding_path,
    vocabulary: Set[str] = None,
    lowercase: bool = False,
    fp: int = 32,
    block_size: int = 665536,
    filter_words: bool = False,
) -> (List[str], List[np.ndarray]):

    start_time = time.time()
    dtype = get_fp_type_numpy(fp=fp)
    print(f"--> Loading embedding from {embedding_path} dtype={dtype}<--")
    if vocabulary:
        print(
            f"We will load only the {len(vocabulary)} words provided as vocabulary (if they appear in the embedding)"
        )
    if filter_words:
        vocab: Set[str] = get_vocab(
            embedding_path=embedding_path, block_size=block_size
        )
    else:
        vocab: Set[str] = set()

    read_words: set = set()
    with open(embedding_path, "r", encoding="utf8") as f:
        try:
            header: str = f.readline()
            num_words, dimensions = header.split(" ")
            num_words, dimensions = int(num_words), int(dimensions)
        except ValueError as err:
            raise ValueError(
                f"Error reading header. "
                f"Expecting embedding in the word2vec format. "
                f"Header expected: num_words dims. Header found: {header}.\n"
                f"Error: {err}"
            )

        lines: List[str] = f.readlines(block_size)

        words_filtered: List[str] = []
        vectors_filtered: List[np.ndarray] = []
        with tqdm(total=num_words, desc="Loading Embedding:", leave=False) as pbar:
            while lines:

                words, vectors = format_lines(
                    lowercase=lowercase,
                    dimensions=dimensions,
                    vocabulary=vocabulary,
                    dtype=dtype,
                    data=lines,
                )

                for word, vector in zip(words, vectors):
                    if vocabulary is None and filter_words:
                        if not valid_word(word=word, vocab=vocab):
                            continue
                    if word in read_words:
                        logging.warning(f"Duplicated word. Skipping line. {word}...")
                        continue

                    words_filtered.append(word)
                    read_words.add(word)
                    vectors_filtered.append(vector)

                pbar.update(len(words))
                lines: List[str] = f.readlines(block_size)

    print(
        f"{len(words_filtered)} words were read from file {embedding_path} in {round(time.time()-start_time,2)} seconds"
    )

    return words_filtered, vectors_filtered
