try:
    import cupy as cp
    import subprocess as sp
    import os
    import math

    mempool = cp.cuda.MemoryPool()
    cp.cuda.set_allocator(mempool.malloc)
    pinned_mempool = cp.cuda.PinnedMemoryPool()
    cp.cuda.set_pinned_memory_allocator(pinned_mempool.malloc)

    cupy_available = True
    print("Cupy found, we will use CUDA to perform matrix operations")


except ImportError:
    cupy_available = False

    print(
        "[WARNING] Cupy not available, "
        "we will use CPU to perform matrix operations (very slow), "
        "using a GPU is highly recommended. "
    )


import numpy as np
from typing import List, Sized, Iterable, Callable, Union, Tuple
from tqdm import tqdm


def get_gpu_memory() -> int:
    if not cupy_available:
        return 0
    _output_to_list: Callable = lambda x: x.decode("ascii").split("\n")[:-1]
    command: str = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info: List[str] = _output_to_list(sp.check_output(command.split()))[1:]
    memory_free_values: List[int] = [
        int(x.split()[0]) for i, x in enumerate(memory_free_info)
    ]
    return int(memory_free_values[0] * 1.049)  # MiB to MB


def get_batch_size(samples: List[np.ndarray], mem_percentage: float = 0.8) -> int:
    # Dynamic batching
    assert cupy_available, "Cupy not available, unable to use GPU"
    assert len(samples) > 0

    samples_size: int = 0  # bytes
    for sample in samples:
        initial_bytes: int = mempool.used_bytes()
        a: cp.ndarray = cp.array(sample)
        final_bytes: int = mempool.used_bytes()
        samples_size += final_bytes - initial_bytes
        del a

    total_memory: int = get_gpu_memory() - mempool.total_bytes()
    batch_size = math.floor((total_memory / samples_size) * mem_percentage)
    print(
        f"[Dynamic Batching] Samples_size {samples_size} bytes. "
        f"Available GPU memory {total_memory} bytes. "
        f"Batch size: {batch_size}."
    )

    assert batch_size > 0, "Error not enough GPU memory for this operation"

    return batch_size


def batch(iterable: Sized, n: int = 1) -> Iterable:
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


def dot(
    a: np.ndarray,
    b: np.ndarray,
    batch_size: int = 10000,
    force_cpu: bool = False,
    show_progress: bool = False,
) -> np.ndarray:

    assert (
        a.shape[1] == b.shape[1]
    ), f"Both matrices should have the number of dimensions. A {a.shape}. B {b.shape}"

    if force_cpu or not cupy_available:
        return a.dot(b.T)
    else:

        result: np.ndarray = np.zeros(
            (a.shape[0], b.shape[0]), dtype=np.result_type(a.dtype, b.dtype)
        )

        for a_index in (
            tqdm(range(0, a.shape[0], batch_size), desc="Dot product")
            if show_progress
            else range(0, a.shape[0], batch_size)
        ):
            for b_index in range(0, b.shape[0], batch_size):
                a_end = min(a_index + batch_size, a.shape[0])
                b_end = min(b_index + batch_size, b.shape[0])
                result[a_index:a_end, b_index:b_end] = cp.asnumpy(
                    cp.asarray(a[a_index:a_end]).dot(cp.asarray(b[b_index:b_end]).T)
                )

        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

        return result


def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
    batch_size: int = 10000,
    force_cpu: bool = False,
    show_progress: bool = False,
) -> np.ndarray:

    assert (
        a.shape[1] == b.shape[1]
    ), f"Both matrices should have the number of dimensions. A {a.shape}. B {b.shape}"

    if force_cpu or not cupy_available:
        norms: np.ndarray = np.sqrt(np.sum(a ** 2, axis=1))
        norms[norms == 0] = 1
        na: np.ndarray = a / norms[:, np.newaxis]

        norms: np.ndarray = np.sqrt(np.sum(b ** 2, axis=1))
        norms[norms == 0] = 1
        nb: np.ndarray = b / norms[:, np.newaxis]

        return na.dot(nb.T)

    else:

        result: np.ndarray = np.zeros(
            (a.shape[0], b.shape[0]), dtype=np.result_type(a.dtype, b.dtype)
        )

        for a_index in (
            tqdm(range(0, a.shape[0], batch_size), desc="Dot product")
            if show_progress
            else range(0, a.shape[0], batch_size)
        ):
            for b_index in range(0, b.shape[0], batch_size):
                a_end = min(a_index + batch_size, a.shape[0])
                b_end = min(b_index + batch_size, b.shape[0])

                a_batch = cp.asarray(a[a_index:a_end])
                b_batch = cp.asarray(b[b_index:b_end])

                norms = cp.sqrt(cp.sum(a_batch ** 2, axis=1))
                norms[norms == 0] = 1
                a_batch /= norms[:, cp.newaxis]

                norms = cp.sqrt(cp.sum(b_batch ** 2, axis=1))
                b_batch /= norms[:, cp.newaxis]

                result[a_index:a_end, b_index:b_end] = cp.asnumpy(
                    a_batch.dot(b_batch.T)
                )

        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

        return result


def dot_rowwise(
    a: np.ndarray,
    b: np.ndarray,
    batch_size: int = 10000,
    force_cpu: bool = False,
    show_progress: bool = False,
) -> np.ndarray:

    assert (
        a.shape == b.shape
    ), f"Both matrices should have the same shape. A {a.shape}. B {b.shape}"

    if force_cpu or not cupy_available:
        return np.einsum("ij,ij->i", a, b)
    else:

        result: np.ndarray = np.zeros(
            (a.shape[0]), dtype=np.result_type(a.dtype, b.dtype)
        )

        for index in (
            tqdm(range(0, a.shape[0], batch_size), desc="row wise dot product")
            if show_progress
            else range(0, a.shape[0], batch_size)
        ):
            end: int = min(index + batch_size, a.shape[0])
            result[index:end] = cp.asnumpy(
                cp.einsum(
                    "ij,ij->i", cp.asarray(a[index:end]), cp.asarray(b[index:end])
                )
            )

        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

        return result


def cosine_rowwise(
    a: np.ndarray,
    b: np.ndarray,
    batch_size: int = 10000,
    force_cpu: bool = False,
    show_progress: bool = False,
) -> np.ndarray:

    assert (
        a.shape == b.shape
    ), f"Both matrices should have the same shape. A {a.shape}. B {b.shape}"

    if force_cpu or not cupy_available:

        norms: np.ndarray = np.sqrt(np.sum(a ** 2, axis=1))
        norms[norms == 0] = 1
        na: np.ndarray = a / norms[:, np.newaxis]

        norms: np.ndarray = np.sqrt(np.sum(b ** 2, axis=1))
        norms[norms == 0] = 1
        nb: np.ndarray = b / norms[:, np.newaxis]

        return np.einsum("ij,ij->i", na, nb)

    else:

        result: np.ndarray = np.zeros(
            (a.shape[0]), dtype=np.result_type(a.dtype, b.dtype)
        )

        for index in (
            tqdm(range(0, a.shape[0], batch_size), desc="row wise dot product")
            if show_progress
            else range(0, a.shape[0], batch_size)
        ):
            end: int = min(index + batch_size, a.shape[0])

            a_batch = cp.asarray(a[index:end])
            b_batch = cp.asarray(b[index:end])

            norms = cp.sqrt(cp.sum(a_batch ** 2, axis=1))
            norms[norms == 0] = 1
            a_batch /= norms[:, cp.newaxis]

            norms = cp.sqrt(cp.sum(b_batch ** 2, axis=1))
            b_batch /= norms[:, cp.newaxis]

            result[index:end] = cp.asnumpy(
                cp.einsum("ij,ij->i", cp.asarray(a_batch), cp.asarray(b_batch))
            )

        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

        return result


def squared_distance(
    a: np.ndarray,
    b: np.ndarray,
    batch_size: int = 10000,
    force_cpu: bool = False,
    show_progress: bool = False,
) -> np.ndarray:

    assert (
        a.shape[1] == b.shape[1]
    ), f"Both matrices should have the number of dimensions. A {a.shape}. B {b.shape}"

    if force_cpu or not cupy_available:
        return np.sum((b[np.newaxis, :] - a[:, np.newaxis]) ** 2, -1)
    else:

        result: np.ndarray = np.zeros(
            (a.shape[0], b.shape[0]), dtype=np.result_type(a.dtype, b.dtype)
        )

        for a_index in (
            tqdm(range(0, a.shape[0], batch_size), desc="Squared distance")
            if show_progress
            else range(0, a.shape[0], batch_size)
        ):
            for b_index in range(0, b.shape[0], batch_size):
                a_end = min(a_index + batch_size, a.shape[0])
                b_end = min(b_index + batch_size, b.shape[0])
                result[a_index:a_end, b_index:b_end] = cp.asnumpy(
                    cp.sum(
                        (
                            cp.asarray(b[b_index:b_end][np.newaxis, :])
                            - cp.asarray(a[a_index:a_end][:, np.newaxis])
                        )
                        ** 2,
                        axis=-1,
                    )
                )

        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

        return result


def squared_distance_rowwise(
    a: np.ndarray,
    b: np.ndarray,
    batch_size: int = 10000,
    force_cpu: bool = False,
    show_progress: bool = False,
) -> np.ndarray:

    assert (
        a.shape == b.shape
    ), f"Both matrices should have the same shape. A {a.shape}. B {b.shape}"

    if force_cpu or not cupy_available:
        return np.sum((b - a) ** 2, axis=-1)
    else:

        result: np.ndarray = np.zeros(
            (a.shape[0]), dtype=np.result_type(a.dtype, b.dtype)
        )

        for index in (
            tqdm(range(0, a.shape[0], batch_size), desc="Row wise squared distance")
            if show_progress
            else range(0, a.shape[0], batch_size)
        ):
            end: int = min(index + batch_size, a.shape[0])
            result[index:end] = cp.asnumpy(
                cp.sum(
                    (cp.asarray(b[index:end]) - cp.asarray(a[index:end])) ** 2,
                    axis=-1,
                )
            )

        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

        return result


def euclidean_distance(
    a: np.ndarray,
    b: np.ndarray,
    batch_size: int = 10000,
    force_cpu: bool = False,
    show_progress: bool = False,
) -> np.ndarray:

    assert (
        a.shape[1] == b.shape[1]
    ), f"Both matrices should have the number of dimensions. A {a.shape}. B {b.shape}"

    if force_cpu or not cupy_available:
        return np.sqrt(np.sum((b[np.newaxis, :] - a[:, np.newaxis]) ** 2, -1))
    else:

        result: np.ndarray = np.zeros(
            (a.shape[0], b.shape[0]), dtype=np.result_type(a.dtype, b.dtype)
        )

        for a_index in (
            tqdm(range(0, a.shape[0], batch_size), desc="Euclidean distance")
            if show_progress
            else range(0, a.shape[0], batch_size)
        ):
            for b_index in range(0, b.shape[0], batch_size):
                a_end = min(a_index + batch_size, a.shape[0])
                b_end = min(b_index + batch_size, b.shape[0])
                result[a_index:a_end, b_index:b_end] = cp.asnumpy(
                    cp.sqrt(
                        cp.sum(
                            (
                                cp.asarray(b[b_index:b_end][np.newaxis, :])
                                - cp.asarray(a[a_index:a_end][:, np.newaxis])
                            )
                            ** 2,
                            axis=-1,
                        )
                    )
                )

        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

        return result


def euclidean_distance_rowwise(
    a: np.ndarray,
    b: np.ndarray,
    batch_size: int = 10000,
    force_cpu: bool = False,
    show_progress: bool = False,
) -> np.ndarray:

    assert (
        a.shape == b.shape
    ), f"Both matrices should have the same shape. A {a.shape}. B {b.shape}"

    if force_cpu or not cupy_available:
        return np.sqrt(np.sum((b - a) ** 2, axis=-1))
    else:

        result: np.ndarray = np.zeros(
            (a.shape[0]), dtype=np.result_type(a.dtype, b.dtype)
        )

        for index in (
            tqdm(range(0, a.shape[0], batch_size), "Row wise euclidean distance")
            if show_progress
            else range(0, a.shape[0], batch_size)
        ):
            end: int = min(index + batch_size, a.shape[0])
            result[index:end] = cp.asnumpy(
                cp.sqrt(
                    cp.sum(
                        (cp.asarray(b[index:end]) - cp.asarray(a[index:end])) ** 2,
                        axis=-1,
                    )
                )
            )

        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

        return result


def knn_dot(
    a: np.ndarray,
    b: np.ndarray,
    k: int = 1,
    batch_size: int = 10000,
    force_cpu: bool = False,
    show_progress: bool = False,
    ordered: bool = False,
    return_distances=False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    # If ordered == True the result will be ordered by the dot product (O(n)),
    # else we will return the k-nn in a random order (O(nlog(n))

    assert (
        a.shape[1] == b.shape[1]
    ), f"Both matrices should have the number of dimensions. A {a.shape}. B {b.shape}"

    assert (
        b.shape[0] >= k
    ), f"K cannot be larger than the number of neighborhoods. k:{k}. Number of neighborhoods: {b.shape[0]}"

    result: np.ndarray = np.zeros((a.shape[0], k), dtype=np.int32)

    if return_distances:
        result_distances: np.ndarray = np.zeros(
            (a.shape[0], k), dtype=np.result_type(a.dtype, b.dtype)
        )
    if force_cpu or not cupy_available:

        for a_index in (
            tqdm(range(0, a.shape[0], batch_size), desc="Dot product KNN")
            if show_progress
            else range(0, a.shape[0], batch_size)
        ):

            a_end = min(a_index + batch_size, a.shape[0])

            batch_result_indexes: np.ndarray = np.zeros(
                (a_end - a_index, k),
                dtype=np.int32,
            )
            batch_result_distances: np.ndarray = np.full(
                (a_end - a_index, k),
                fill_value=np.float("-inf"),
                dtype=np.result_type(a.dtype, b.dtype),
            )

            for b_index in range(0, b.shape[0], batch_size):
                b_end = min(b_index + batch_size, b.shape[0])
                distances = a[a_index:a_end].dot(b[b_index:b_end].T)

                batch_result_indexes2 = np.argpartition(
                    distances, kth=-min(k, distances.shape[1]), axis=1
                )[:, -k:]
                batch_result_distances2 = distances[
                    np.arange(distances.shape[0])[:, np.newaxis], batch_result_indexes2
                ]

                new_batch_result_distances = np.concatenate(
                    (batch_result_distances, batch_result_distances2), axis=1
                )
                new_batch_result_indexes = np.concatenate(
                    (batch_result_indexes, batch_result_indexes2 + b_index), axis=1
                )
                new_indexes = np.argpartition(
                    new_batch_result_distances, kth=-k, axis=1
                )[:, -k:]
                batch_result_indexes = new_batch_result_indexes[
                    np.arange(new_batch_result_indexes.shape[0])[:, np.newaxis],
                    new_indexes,
                ]

                batch_result_distances = new_batch_result_distances[
                    np.arange(new_batch_result_distances.shape[0])[:, np.newaxis],
                    new_indexes,
                ]

            if not ordered:
                result[a_index:a_end] = batch_result_indexes
                if return_distances:
                    result_distances[a_index:a_end] = batch_result_distances

            else:
                indexes = np.flip(
                    batch_result_distances.argsort(axis=1),
                    axis=1,
                )

                result[a_index:a_end] = batch_result_indexes[
                    np.arange(batch_result_indexes.shape[0])[:, None], indexes
                ]

                if return_distances:
                    result_distances[a_index:a_end] = batch_result_distances[
                        np.arange(batch_result_distances.shape[0])[:, None], indexes
                    ]

    else:

        for a_index in (
            tqdm(range(0, a.shape[0], batch_size), desc="Dot product KNN")
            if show_progress
            else range(0, a.shape[0], batch_size)
        ):

            a_end = min(a_index + batch_size, a.shape[0])

            batch_result_indexes: cp.ndarray = cp.zeros(
                (a_end - a_index, k),
                dtype=cp.int32,
            )
            batch_result_distances: cp.ndarray = cp.full(
                (a_end - a_index, k),
                fill_value=np.float("-inf"),
                dtype=np.result_type(a.dtype, b.dtype),
            )

            for b_index in range(0, b.shape[0], batch_size):
                b_end = min(b_index + batch_size, b.shape[0])
                distances = cp.asarray(a[a_index:a_end]).dot(
                    cp.asarray(b[b_index:b_end]).T
                )

                batch_result_indexes2 = cp.argpartition(
                    distances, kth=-min(k, distances.shape[1]), axis=1
                )[:, -k:]
                batch_result_distances2 = distances[
                    cp.arange(distances.shape[0])[:, cp.newaxis], batch_result_indexes2
                ]

                new_batch_result_distances = cp.concatenate(
                    (batch_result_distances, batch_result_distances2), axis=1
                )
                new_batch_result_indexes = cp.concatenate(
                    (batch_result_indexes, batch_result_indexes2 + b_index), axis=1
                )
                new_indexes = cp.argpartition(
                    new_batch_result_distances, kth=-k, axis=1
                )[:, -k:]
                batch_result_indexes = new_batch_result_indexes[
                    cp.arange(new_batch_result_indexes.shape[0])[:, cp.newaxis],
                    new_indexes,
                ]

                batch_result_distances = new_batch_result_distances[
                    cp.arange(new_batch_result_distances.shape[0])[:, cp.newaxis],
                    new_indexes,
                ]

            if not ordered:
                result[a_index:a_end] = cp.asnumpy(batch_result_indexes)
                if return_distances:
                    result_distances[a_index:a_end] = cp.asnumpy(batch_result_distances)

            else:
                indexes = cp.flip(
                    batch_result_distances.argsort(axis=1),
                    axis=1,
                )

                result[a_index:a_end] = cp.asnumpy(
                    batch_result_indexes[
                        np.arange(batch_result_indexes.shape[0])[:, None], indexes
                    ]
                )
                if return_distances:
                    result_distances[a_index:a_end] = cp.asnumpy(
                        batch_result_distances[
                            np.arange(batch_result_distances.shape[0])[:, None], indexes
                        ]
                    )

        del batch_result_indexes
        del batch_result_distances

        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

    if return_distances:
        return result, result_distances
    else:
        return result


def knn_cosine(
    a: np.ndarray,
    b: np.ndarray,
    k: int = 1,
    batch_size: int = 10000,
    force_cpu: bool = False,
    show_progress: bool = False,
    ordered: bool = False,
    return_distances=False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    # If ordered == True the result will be ordered by the dot product (O(n)),
    # else we will return the k-nn in a random order (O(nlog(n))

    assert (
        a.shape[1] == b.shape[1]
    ), f"Both matrices should have the number of dimensions. A {a.shape}. B {b.shape}"

    assert (
        b.shape[0] >= k
    ), f"K cannot be larger than the number of neighborhoods. k:{k}. Number of neighborhoods: {b.shape[0]}"

    result: np.ndarray = np.zeros((a.shape[0], k), dtype=np.int32)

    if return_distances:
        result_distances: np.ndarray = np.zeros(
            (a.shape[0], k), dtype=np.result_type(a.dtype, b.dtype)
        )
    if force_cpu or not cupy_available:

        for a_index in (
            tqdm(range(0, a.shape[0], batch_size), desc="Dot product KNN")
            if show_progress
            else range(0, a.shape[0], batch_size)
        ):

            a_end = min(a_index + batch_size, a.shape[0])

            batch_result_indexes: np.ndarray = np.zeros(
                (a_end - a_index, k),
                dtype=np.int32,
            )
            batch_result_distances: np.ndarray = np.full(
                (a_end - a_index, k),
                fill_value=np.float("-inf"),
                dtype=np.result_type(a.dtype, b.dtype),
            )

            for b_index in range(0, b.shape[0], batch_size):
                b_end = min(b_index + batch_size, b.shape[0])
                a_batch = a[a_index:a_end]
                b_batch = b[b_index:b_end]

                norms = np.sqrt(np.sum(a_batch ** 2, axis=1))
                norms[norms == 0] = 1
                a_batch /= norms[:, np.newaxis]

                norms = np.sqrt(np.sum(b_batch ** 2, axis=1))
                b_batch /= norms[:, np.newaxis]

                distances = a_batch.dot(b_batch.T)

                batch_result_indexes2 = np.argpartition(
                    distances, kth=-min(k, distances.shape[1]), axis=1
                )[:, -k:]
                batch_result_distances2 = distances[
                    np.arange(distances.shape[0])[:, np.newaxis], batch_result_indexes2
                ]

                new_batch_result_distances = np.concatenate(
                    (batch_result_distances, batch_result_distances2), axis=1
                )
                new_batch_result_indexes = np.concatenate(
                    (batch_result_indexes, batch_result_indexes2 + b_index), axis=1
                )
                new_indexes = np.argpartition(
                    new_batch_result_distances, kth=-k, axis=1
                )[:, -k:]
                batch_result_indexes = new_batch_result_indexes[
                    np.arange(new_batch_result_indexes.shape[0])[:, np.newaxis],
                    new_indexes,
                ]

                batch_result_distances = new_batch_result_distances[
                    np.arange(new_batch_result_distances.shape[0])[:, np.newaxis],
                    new_indexes,
                ]

            if not ordered:
                result[a_index:a_end] = batch_result_indexes
                if return_distances:
                    result_distances[a_index:a_end] = batch_result_distances

            else:
                indexes = np.flip(
                    batch_result_distances.argsort(axis=1),
                    axis=1,
                )

                result[a_index:a_end] = batch_result_indexes[
                    np.arange(batch_result_indexes.shape[0])[:, None], indexes
                ]

                if return_distances:
                    result_distances[a_index:a_end] = batch_result_distances[
                        np.arange(batch_result_distances.shape[0])[:, None], indexes
                    ]

    else:

        for a_index in (
            tqdm(range(0, a.shape[0], batch_size), desc="Dot product KNN")
            if show_progress
            else range(0, a.shape[0], batch_size)
        ):

            a_end = min(a_index + batch_size, a.shape[0])

            batch_result_indexes: cp.ndarray = cp.zeros(
                (a_end - a_index, k),
                dtype=cp.int32,
            )
            batch_result_distances: cp.ndarray = cp.full(
                (a_end - a_index, k),
                fill_value=np.float("-inf"),
                dtype=np.result_type(a.dtype, b.dtype),
            )

            for b_index in range(0, b.shape[0], batch_size):
                b_end = min(b_index + batch_size, b.shape[0])

                a_batch = cp.asarray(a[a_index:a_end])
                b_batch = cp.asarray(b[b_index:b_end])

                norms = cp.sqrt(cp.sum(a_batch ** 2, axis=1))
                norms[norms == 0] = 1
                a_batch /= norms[:, cp.newaxis]

                norms = cp.sqrt(cp.sum(b_batch ** 2, axis=1))
                b_batch /= norms[:, cp.newaxis]

                distances = a_batch.dot(b_batch.T)

                batch_result_indexes2 = cp.argpartition(
                    distances, kth=-min(k, distances.shape[1]), axis=1
                )[:, -k:]
                batch_result_distances2 = distances[
                    cp.arange(distances.shape[0])[:, cp.newaxis], batch_result_indexes2
                ]

                new_batch_result_distances = cp.concatenate(
                    (batch_result_distances, batch_result_distances2), axis=1
                )
                new_batch_result_indexes = cp.concatenate(
                    (batch_result_indexes, batch_result_indexes2 + b_index), axis=1
                )
                new_indexes = cp.argpartition(
                    new_batch_result_distances, kth=-k, axis=1
                )[:, -k:]
                batch_result_indexes = new_batch_result_indexes[
                    cp.arange(new_batch_result_indexes.shape[0])[:, cp.newaxis],
                    new_indexes,
                ]

                batch_result_distances = new_batch_result_distances[
                    cp.arange(new_batch_result_distances.shape[0])[:, cp.newaxis],
                    new_indexes,
                ]

            if not ordered:
                result[a_index:a_end] = cp.asnumpy(batch_result_indexes)
                if return_distances:
                    result_distances[a_index:a_end] = cp.asnumpy(batch_result_distances)

            else:
                indexes = cp.flip(
                    batch_result_distances.argsort(axis=1),
                    axis=1,
                )

                result[a_index:a_end] = cp.asnumpy(
                    batch_result_indexes[
                        np.arange(batch_result_indexes.shape[0])[:, None], indexes
                    ]
                )
                if return_distances:
                    result_distances[a_index:a_end] = cp.asnumpy(
                        batch_result_distances[
                            np.arange(batch_result_distances.shape[0])[:, None], indexes
                        ]
                    )

        del batch_result_indexes
        del batch_result_distances

        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

    if return_distances:
        return result, result_distances
    else:
        return result


def knn_euclidean_distance(
    a: np.ndarray,
    b: np.ndarray,
    k: int = 1,
    batch_size: int = 10000,
    force_cpu: bool = False,
    show_progress: bool = False,
    ordered: bool = False,
    return_distances=False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    # If ordered == True the result will be ordered by the dot product (O(n)),
    # else we will return the k-nn in a random order (O(nlog(n))

    assert (
        a.shape[1] == b.shape[1]
    ), f"Both matrices should have the number of dimensions. A {a.shape}. B {b.shape}"

    assert (
        b.shape[0] >= k
    ), f"K cannot be larger than the number of neighborhoods. k:{k}. Number of neighborhoods: {b.shape[0]}"

    result: np.ndarray = np.zeros((a.shape[0], k), dtype=np.int32)

    if return_distances:
        result_distances: np.ndarray = np.zeros(
            (a.shape[0], k), dtype=np.result_type(a.dtype, b.dtype)
        )
    if force_cpu or not cupy_available:

        for a_index in (
            tqdm(range(0, a.shape[0], batch_size), desc="Euclidean distance KNN")
            if show_progress
            else range(0, a.shape[0], batch_size)
        ):

            a_end = min(a_index + batch_size, a.shape[0])

            batch_result_indexes: np.ndarray = np.zeros(
                (a_end - a_index, k),
                dtype=np.int32,
            )
            batch_result_distances: np.ndarray = np.full(
                (a_end - a_index, k),
                fill_value=np.float("inf"),
                dtype=np.result_type(a.dtype, b.dtype),
            )

            for b_index in range(0, b.shape[0], batch_size):
                b_end = min(b_index + batch_size, b.shape[0])
                distances = np.sum(
                    (b[b_index:b_end][np.newaxis, :] - a[a_index:a_end][:, np.newaxis])
                    ** 2,
                    -1,
                )

                batch_result_indexes2 = np.argpartition(
                    distances, kth=min(k, distances.shape[1]), axis=1
                )[:, :k]
                batch_result_distances2 = distances[
                    np.arange(distances.shape[0])[:, np.newaxis], batch_result_indexes2
                ]

                new_batch_result_distances = np.concatenate(
                    (batch_result_distances, batch_result_distances2), axis=1
                )
                new_batch_result_indexes = np.concatenate(
                    (batch_result_indexes, batch_result_indexes2 + b_index), axis=1
                )
                new_indexes = np.argpartition(
                    new_batch_result_distances, kth=k, axis=1
                )[:, :k]
                batch_result_indexes = new_batch_result_indexes[
                    np.arange(new_batch_result_indexes.shape[0])[:, np.newaxis],
                    new_indexes,
                ]

                batch_result_distances = new_batch_result_distances[
                    np.arange(new_batch_result_distances.shape[0])[:, np.newaxis],
                    new_indexes,
                ]

            if not ordered:
                result[a_index:a_end] = batch_result_indexes
                if return_distances:
                    result_distances[a_index:a_end] = batch_result_distances

            else:
                indexes = batch_result_distances.argsort(axis=1)

                result[a_index:a_end] = batch_result_indexes[
                    np.arange(batch_result_indexes.shape[0])[:, None], indexes
                ]

                if return_distances:
                    result_distances[a_index:a_end] = batch_result_distances[
                        np.arange(batch_result_distances.shape[0])[:, None], indexes
                    ]

    else:

        for a_index in (
            tqdm(range(0, a.shape[0], batch_size), desc="Dot product KNN")
            if show_progress
            else range(0, a.shape[0], batch_size)
        ):

            a_end = min(a_index + batch_size, a.shape[0])

            batch_result_indexes: cp.ndarray = cp.zeros(
                (a_end - a_index, k),
                dtype=cp.int32,
            )
            batch_result_distances: cp.ndarray = cp.full(
                (a_end - a_index, k),
                fill_value=np.float("inf"),
                dtype=np.result_type(a.dtype, b.dtype),
            )

            for b_index in range(0, b.shape[0], batch_size):
                b_end = min(b_index + batch_size, b.shape[0])
                distances = cp.sum(
                    (
                        cp.asarray(b[b_index:b_end][np.newaxis, :])
                        - cp.asarray(a[a_index:a_end][:, np.newaxis])
                        - cp.asarray(a[a_index:a_end][:, np.newaxis])
                    )
                    ** 2,
                    -1,
                )

                batch_result_indexes2 = cp.argpartition(
                    distances, kth=min(k, distances.shape[1]), axis=1
                )[:, :k]
                batch_result_distances2 = distances[
                    cp.arange(distances.shape[0])[:, cp.newaxis], batch_result_indexes2
                ]

                new_batch_result_distances = cp.concatenate(
                    (batch_result_distances, batch_result_distances2), axis=1
                )
                new_batch_result_indexes = cp.concatenate(
                    (batch_result_indexes, batch_result_indexes2 + b_index), axis=1
                )
                new_indexes = cp.argpartition(
                    new_batch_result_distances, kth=k, axis=1
                )[:, :k]
                batch_result_indexes = new_batch_result_indexes[
                    cp.arange(new_batch_result_indexes.shape[0])[:, cp.newaxis],
                    new_indexes,
                ]

                batch_result_distances = new_batch_result_distances[
                    cp.arange(new_batch_result_distances.shape[0])[:, cp.newaxis],
                    new_indexes,
                ]

            if not ordered:
                result[a_index:a_end] = cp.asnumpy(batch_result_indexes)
                if return_distances:
                    result_distances[a_index:a_end] = cp.asnumpy(batch_result_distances)

            else:
                indexes = batch_result_distances.argsort(axis=1)

                result[a_index:a_end] = cp.asnumpy(
                    batch_result_indexes[
                        np.arange(batch_result_indexes.shape[0])[:, None], indexes
                    ]
                )
                if return_distances:
                    result_distances[a_index:a_end] = cp.asnumpy(
                        batch_result_distances[
                            np.arange(batch_result_distances.shape[0])[:, None], indexes
                        ]
                    )

        del batch_result_indexes
        del batch_result_distances

        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

    if return_distances:
        return result, result_distances
    else:
        return result


def length_normalize(
    matrix: np.ndarray,
    force_cpu=False,
    show_progress: bool = False,
    batch_size: int = 10000,
    inplace=True,
) -> np.ndarray:

    if not inplace:
        new_matrix: np.ndarray = np.zeros(
            (matrix.shape[0], matrix.shape[1]), dtype=matrix.dtype
        )

    if not cupy_available or force_cpu:
        norms = np.sqrt(np.sum(matrix ** 2, axis=1))
        norms[norms == 0] = 1
        if inplace:
            matrix = matrix / norms[:, np.newaxis]
        else:
            new_matrix = matrix / norms[:, np.newaxis]

    else:
        for m_index in (
            tqdm(range(0, matrix.shape[0], batch_size), desc="Normalize length")
            if show_progress
            else range(0, matrix.shape[0], batch_size)
        ):
            m_end = min(m_index + batch_size, matrix.shape[0])
            batch_matrix = cp.asarray(matrix[m_index:m_end])
            norms = cp.sqrt(cp.sum(batch_matrix ** 2, axis=1))
            norms[norms == 0] = 1
            if inplace:
                matrix[m_index:m_end] = cp.asnumpy(batch_matrix / norms[:, np.newaxis])
            else:
                new_matrix[m_index:m_end] = cp.asnumpy(
                    batch_matrix / norms[:, np.newaxis]
                )

        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

    if inplace:
        return matrix
    else:
        return new_matrix


def length_normalize_dimensionwise(
    matrix: np.ndarray,
    force_cpu=False,
    show_progress: bool = False,
    batch_size: int = 10000,
    inplace=True,
) -> np.ndarray:

    if not inplace:
        new_matrix: np.ndarray = np.zeros(
            (matrix.shape[0], matrix.shape[1]), dtype=matrix.dtype
        )

    if not cupy_available or force_cpu:
        norms = np.sqrt(np.sum(matrix ** 2, axis=0))
        norms[norms == 0] = 1
        if inplace:
            matrix = matrix / norms
        else:
            new_matrix = matrix / norms

    else:
        for m_index in (
            tqdm(
                range(0, matrix.shape[1], batch_size),
                desc="Normalize length dimension wise",
            )
            if show_progress
            else range(0, matrix.shape[1], batch_size)
        ):
            m_end = min(m_index + batch_size, matrix.shape[1])
            batch_matrix = cp.asarray(matrix[:, m_index:m_end])
            norms = cp.sqrt(cp.sum(batch_matrix ** 2, axis=0))
            norms[norms == 0] = 1
            if inplace:
                matrix[:, m_index:m_end] = cp.asnumpy(batch_matrix / norms)
            else:
                new_matrix[:, m_index:m_end] = cp.asnumpy(batch_matrix / norms)

        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

    if inplace:
        return matrix
    else:
        return new_matrix


def l1_length_normalize(
    matrix: np.ndarray,
    force_cpu=False,
    show_progress: bool = False,
    batch_size: int = 10000,
    inplace=True,
) -> np.ndarray:

    if not inplace:
        new_matrix: np.ndarray = np.zeros(
            (matrix.shape[0], matrix.shape[1]), dtype=matrix.dtype
        )

    if not cupy_available or force_cpu:
        norms = np.sum(matrix, axis=1)
        norms[norms == 0] = 1
        if inplace:
            matrix = matrix / norms[:, np.newaxis]
        else:
            new_matrix = matrix / norms[:, np.newaxis]

    else:
        for m_index in (
            tqdm(
                range(0, matrix.shape[0], batch_size), desc="Normalize length (L1 norm)"
            )
            if show_progress
            else range(0, matrix.shape[0], batch_size)
        ):
            m_end = min(m_index + batch_size, matrix.shape[0])
            batch_matrix = cp.asarray(matrix[m_index:m_end])
            norms = cp.sum(batch_matrix, axis=1)
            norms[norms == 0] = 1
            if inplace:
                matrix[m_index:m_end] = cp.asnumpy(batch_matrix / norms[:, np.newaxis])
            else:
                new_matrix[m_index:m_end] = cp.asnumpy(
                    batch_matrix / norms[:, np.newaxis]
                )

        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

    if inplace:
        return matrix
    else:
        return new_matrix


def l1_length_normalize_dimensionwise(
    matrix: np.ndarray,
    force_cpu=False,
    show_progress: bool = False,
    batch_size: int = 10000,
    inplace=True,
) -> np.ndarray:

    if not inplace:
        new_matrix: np.ndarray = np.zeros(
            (matrix.shape[0], matrix.shape[1]), dtype=matrix.dtype
        )

    if not cupy_available or force_cpu:
        norms = np.sum(matrix, axis=0)
        norms[norms == 0] = 1
        if inplace:
            matrix = matrix / norms
        else:
            new_matrix = matrix / norms

    else:
        for m_index in (
            tqdm(
                range(0, matrix.shape[1], batch_size),
                desc="Normalize length dimension wise (L1 norm)",
            )
            if show_progress
            else range(0, matrix.shape[1], batch_size)
        ):
            m_end = min(m_index + batch_size, matrix.shape[1])
            batch_matrix = cp.asarray(matrix[:, m_index:m_end])
            norms = cp.sum(batch_matrix, axis=0)
            norms[norms == 0] = 1
            if inplace:
                matrix[:, m_index:m_end] = cp.asnumpy(batch_matrix / norms)
            else:
                new_matrix[:, m_index:m_end] = cp.asnumpy(batch_matrix / norms)

        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

    if inplace:
        return matrix
    else:
        return new_matrix


def mean_center(
    matrix: np.ndarray,
    force_cpu=False,
    show_progress: bool = False,
    batch_size: int = 100,
    inplace=True,
) -> np.ndarray:

    if not inplace:
        new_matrix: np.ndarray = np.zeros(
            (matrix.shape[0], matrix.shape[1]), dtype=matrix.dtype
        )

    if not cupy_available or force_cpu:
        avg = np.mean(matrix, axis=0)
        if inplace:
            matrix -= avg
        else:
            new_matrix = matrix - avg

    else:
        for m_index in (
            tqdm(range(0, matrix.shape[1], batch_size), desc="Mean center")
            if show_progress
            else range(0, matrix.shape[1], batch_size)
        ):
            m_end = min(m_index + batch_size, matrix.shape[1])
            batch_matrix = cp.asarray(matrix[:, m_index:m_end])
            avg = cp.mean(batch_matrix, axis=0)
            if inplace:
                matrix[:, m_index:m_end] = cp.asnumpy(batch_matrix - avg)
            else:
                new_matrix[:, m_index:m_end] = cp.asnumpy(batch_matrix - avg)

        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

    if inplace:
        return matrix
    else:
        return new_matrix


def mean_center_rowwise(
    matrix: np.ndarray,
    force_cpu=False,
    show_progress: bool = False,
    batch_size: int = 10000,
    inplace=True,
) -> np.ndarray:

    if not inplace:
        new_matrix: np.ndarray = np.zeros(
            (matrix.shape[0], matrix.shape[1]), dtype=matrix.dtype
        )

    if not cupy_available or force_cpu:
        avg = np.mean(matrix, axis=1)
        if inplace:
            matrix -= avg[:, np.newaxis]
        else:
            new_matrix = matrix - avg[:, np.newaxis]

    else:
        for m_index in (
            tqdm(range(0, matrix.shape[0], batch_size), desc="Mean center row wise")
            if show_progress
            else range(0, matrix.shape[0], batch_size)
        ):
            m_end = min(m_index + batch_size, matrix.shape[0])
            batch_matrix = cp.asarray(matrix[m_index:m_end])
            avg = cp.mean(batch_matrix, axis=1)
            if inplace:
                matrix[m_index:m_end] = cp.asnumpy(batch_matrix - avg[:, np.newaxis])
            else:
                new_matrix[m_index:m_end] = cp.asnumpy(
                    batch_matrix - avg[:, np.newaxis]
                )

        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

    if inplace:
        return matrix
    else:
        return new_matrix


def average_lists(
    matrix_list: List[np.ndarray],
    batch_size: int = 10000,
    force_cpu: bool = False,
    show_progress: bool = False,
) -> np.ndarray:

    shape = set([m.shape for m in matrix_list])
    assert (
        len(shape) == 1
    ), f"All the matrices should have the same shapes. Shapes found: {shape}"

    shape = shape.pop()

    result: np.ndarray = np.zeros(
        (shape[0], shape[1]), dtype=np.result_type([m.dtype for m in matrix_list])
    )

    if not cupy_available or force_cpu:
        for m in matrix_list:
            result += m
        return result / len(matrix_list)

    else:
        for m_index in (
            tqdm(range(0, result.shape[0], batch_size), desc="Average of lists")
            if show_progress
            else range(0, result.shape[0], batch_size)
        ):
            m_end = min(m_index + batch_size, result.shape[0])
            partial_matrix = cp.zeros(
                (m_end - m_index, result.shape[1]), dtype=result.dtype
            )
            for m in matrix_list:
                partial_matrix += cp.asarray(m[m_index:m_end])

            result[m_index:m_end] = cp.asnumpy(partial_matrix / len(matrix_list))

        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

        return result


def average_vectors(
    matrix,
    weights=None,
    batch_size: int = 10000,
    force_cpu: bool = False,
    show_progress: bool = False,
) -> np.ndarray:

    assert len(matrix.shape) == 2, f"2D matrix expected. Matrix shape {matrix.shape}"

    if not cupy_available or force_cpu:

        return np.mean(matrix, axis=0)

    else:
        result: cp.ndarray = cp.zeros((matrix.shape[1]), dtype=matrix.dtype)
        for m_index in (
            tqdm(range(0, matrix.shape[0], batch_size), desc="Average vectors")
            if show_progress
            else range(0, matrix.shape[0], batch_size)
        ):
            m_end = min(m_index + batch_size, matrix.shape[0])
            if weights is None:
                result += cp.sum(
                    cp.asarray(matrix[m_index:m_end]),
                    axis=0,
                )
            else:
                result += cp.sum(
                    cp.asarray(matrix[m_index:m_end])
                    * cp.asarray(weights[m_index:m_end])[:, np.newaxis],
                    axis=0,
                )

        if weights is None:
            result = cp.asnumpy(result / matrix.shape[0])
        else:
            result = cp.asnumpy(result / np.sum(weights))

        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

        return result


def average_3_dimensions(
    matrix: np.ndarray,
    batch_size: int = 10000,
    force_cpu: bool = False,
    show_progress: bool = False,
) -> np.ndarray:

    assert (
        len(matrix.shape) == 3
    ), f"This function only works with 3 dimensional matrices. matrix shape: {matrix.shape}"

    if not cupy_available or force_cpu:
        return np.mean(matrix, axis=1)

    else:

        result: np.ndarray = np.zeros(
            (matrix.shape[0], matrix.shape[2]), dtype=matrix.dtype
        )

        for m_index in (
            tqdm(range(0, result.shape[0], batch_size), desc="Average 3D matrix")
            if show_progress
            else range(0, result.shape[0], batch_size)
        ):
            m_end = min(m_index + batch_size, result.shape[0])
            result[m_index:m_end] = cp.asnumpy(
                cp.mean(cp.asarray(matrix[m_index:m_end]), axis=1)
            )

        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

        return result


def length_normalize_vector(
    vector: np.ndarray,
    force_cpu=True,  # Vectors are usually 300 dimensions, for such a small matrix CPU is faster, we use CPU by default
) -> np.ndarray:

    if not cupy_available or force_cpu:
        norm = np.sqrt(np.sum(vector ** 2))
        if norm == 0:
            return vector
        return vector / norm

    else:
        gpu_vector = cp.asarray(vector)
        norm = cp.sqrt(cp.sum(gpu_vector ** 2))
        if norm == 0:
            return vector
        result = cp.asnumpy(gpu_vector / norm)
        del gpu_vector
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

        return result


def matrix_add(
    a: np.ndarray,
    b: np.ndarray,
    batch_size: int = 10000,
    force_cpu: bool = False,
    show_progress: bool = False,
) -> np.ndarray:

    assert (
        a.shape[1] == b.shape[1]
    ), f"Both matrices should have the number of dimensions. A {a.shape}. B {b.shape}"

    if force_cpu or not cupy_available:
        return np.sum((a, b), axis=0)
    else:

        result: np.ndarray = np.zeros(
            (a.shape[0], a.shape[1]), dtype=np.result_type(a.dtype, b.dtype)
        )

        for a_index in (
            tqdm(range(0, a.shape[0], batch_size), desc="Matrix add")
            if show_progress
            else range(0, a.shape[0], batch_size)
        ):

            a_end = min(a_index + batch_size, a.shape[0])
            result[a_index:a_end] = cp.asnumpy(
                cp.asarray(a[a_index:a_end]) + cp.asarray(b[a_index:a_end])
            )

        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

        return result


def CosAdd(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    matrix: np.ndarray,
    k: int,
    batch_size: int = 10000,
    force_cpu: bool = False,
    show_progress: bool = False,
) -> np.ndarray:

    assert (
        a.shape == b.shape == c.shape
    ), f"a,b and c should have the same shape. a {a.shape} b {b.shape} c {c.shape}"

    assert (
        a.shape[1] == matrix.shape[1]
    ), f"vectors in a b c and matrix should have the same number of dimensions. a {a.shape[1]} matrix {matrix.shape[1]}"

    add: np.ndarray = np.zeros(
        (a.shape[0], a.shape[1]), dtype=np.result_type(a.dtype, b.dtype, c.dtype)
    )

    for a_index in (
        tqdm(range(0, a.shape[0], batch_size), desc="Analogy")
        if show_progress
        else range(0, a.shape[0], batch_size)
    ):

        a_end = min(a_index + batch_size, a.shape[0])

        if force_cpu or not cupy_available:
            add[a_index:a_end] = c[a_index:a_end] - a[a_index:a_end] + b[a_index:a_end]

        else:
            add[a_index:a_end] = cp.asnumpy(
                cp.asarray(b[a_index:a_end])
                - cp.asarray(a[a_index:a_end])
                + cp.asarray(c[a_index:a_end])
            )

    if not (force_cpu or not cupy_available):
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

    add = length_normalize(
        add
    )  # Should not be necessary, but is super fast, so lets ensure that all vectors are normalized

    return knn_dot(
        a=add,
        b=matrix,
        k=k,
        batch_size=batch_size,
        force_cpu=force_cpu,
        show_progress=show_progress,
        ordered=True,
        return_distances=False,
    )


def CosAdd_Sim(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    batch_size: int = 10000,
    force_cpu: bool = False,
    show_progress: bool = False,
) -> np.ndarray:

    assert (
        a.shape == b.shape == c.shape == d.shape
    ), f"a,b,c and d should have the same shape. a {a.shape} b {b.shape} c {c.shape} d {d.shape}"

    add: np.ndarray = np.zeros(
        (a.shape[0], a.shape[1]), dtype=np.result_type(a.dtype, b.dtype, c.dtype)
    )

    for a_index in (
        tqdm(range(0, a.shape[0], batch_size), desc="Analogy")
        if show_progress
        else range(0, a.shape[0], batch_size)
    ):

        a_end = min(a_index + batch_size, a.shape[0])

        if force_cpu or not cupy_available:
            add[a_index:a_end] = c[a_index:a_end] - a[a_index:a_end] + b[a_index:a_end]

        else:
            add[a_index:a_end] = cp.asnumpy(
                cp.asarray(c[a_index:a_end])
                - cp.asarray(a[a_index:a_end])
                + cp.asarray(b[a_index:a_end])
            )

    if not (force_cpu or not cupy_available):
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

    add = length_normalize(add)

    return dot_rowwise(
        a=add,
        b=d,
        batch_size=batch_size,
        force_cpu=force_cpu,
        show_progress=show_progress,
    )


def get_average_length(
    matrix: np.ndarray,
    force_cpu=False,
    show_progress: bool = False,
    batch_size: int = 10000,
) -> float:

    if not cupy_available or force_cpu:
        return np.average(np.linalg.norm(matrix, axis=1))

    else:
        sum_length = 0
        for m_index in (
            tqdm(range(0, matrix.shape[0], batch_size), desc="Get average length")
            if show_progress
            else range(0, matrix.shape[0], batch_size)
        ):
            m_end = min(m_index + batch_size, matrix.shape[0])
            batch_matrix = cp.asarray(matrix[m_index:m_end])
            sum_length += cp.sum(cp.linalg.norm(batch_matrix, axis=1))

        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

        return float(cp.asnumpy(sum_length / matrix.shape[0]))


def scale_matrix(
    matrix: np.ndarray,
    scale_factor: float,
    force_cpu=True,  # Too simple for a GPU to deliver significant speedup
    show_progress: bool = False,
    batch_size: int = 10000,
    inplace=True,
) -> np.ndarray:
    if not inplace:
        new_matrix: np.ndarray = np.zeros(
            (matrix.shape[0], matrix.shape[1]), dtype=matrix.dtype
        )

    if not cupy_available or force_cpu:
        if inplace:
            matrix = matrix * scale_factor
        else:
            new_matrix = matrix * scale_factor

    else:
        for m_index in (
            tqdm(range(0, matrix.shape[0], batch_size), desc="Scale matrix")
            if show_progress
            else range(0, matrix.shape[0], batch_size)
        ):
            m_end = min(m_index + batch_size, matrix.shape[0])
            batch_matrix = cp.asarray(matrix[m_index:m_end])

            if inplace:
                matrix[m_index:m_end] = cp.asnumpy(batch_matrix * scale_factor)
            else:
                new_matrix[m_index:m_end] = cp.asnumpy(batch_matrix * scale_factor)

        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

    if inplace:
        return matrix
    else:
        return new_matrix


def rolling_average(
    a: np.ndarray,
    weights: np.ndarray,
    force_cpu: bool = False,
) -> np.ndarray:

    assert (
        a.shape[0] == weights.shape[0]
    ), f"Both matrices should have the same len. A {a.shape}. B {weights.shape}"

    if force_cpu or not cupy_available:
        return np.true_divide(
            (a * weights[None, :].T).cumsum(axis=0), weights.cumsum()[None, :].T
        )

    else:

        a_cp = cp.asarray(a)
        w_cp = cp.asarray(weights)
        result = cp.true_divide(
            (a_cp * w_cp[None, :].T).cumsum(axis=0), w_cp.cumsum()[None, :].T
        )

        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

        return cp.asnumpy(result)
