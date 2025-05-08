import faiss
import hnswlib
import numpy as np


def faiss_index(
    X: np.ndarray,
    d: int,
    M: int,
    metric: str,
    ef_construction: int,
    ef_search: int,
    Q: np.ndarray,
    k: int,
) -> np.ndarray:
    # Create an index
    index = faiss.IndexHNSWFlat(d, M, metric)

    # Set the parameters for the construction of the graph
    index.hnsw.efConstruction = ef_construction  # Number of neighbors to consider for the construction of the graph

    # Add the database to the index
    index.add(X)  # type: ignore

    # Search for the nearest vector to the query vector
    index.hnsw.efSearch = ef_search
    distances, ids = index.search(Q, k)  # type: ignore

    # print(distances)  # Distances to the nearest vectors
    # print(ids)  # Indexes of the nearest vectors

    return ids


def hnswlib_index(
    X: np.ndarray,
    d: int,
    M: int,
    metric: str,
    ef_construction: int,
    ef_search: int,
    Q: np.ndarray,
    k: int,
) -> np.ndarray:
    # Initialize the index
    p = hnswlib.Index(space=metric, dim=d)

    # Initializing index - the maximum number of elements should be known beforehand!
    p.init_index(max_elements=len(X), ef_construction=ef_construction, M=M)

    # Element insertion (can be called several times):
    p.add_items(X)

    # Controlling the recall by setting ef:
    p.set_ef(ef_search)  # ef should always be > k

    # Query dataset, k - number of the closest elements (returns 2 numpy arrays)
    ids, distances = p.knn_query(Q, k)

    # print(distances)  # Distances to the nearest vectors
    # print(ids)  # Indexes of the nearest vectors

    ### Index parameters are exposed as class properties:
    # print(f"Parameters passed to constructor:  space={p.space}, dim={p.dim}")
    # print(f"Index construction: M={p.M}, ef_construction={p.ef_construction}")
    # print(f"Index size is {p.element_count} and index capacity is {p.max_elements}")
    # print(f"Search speed/quality trade-off parameter: ef={p.ef}")

    return ids


def calculate_average_recall(ids: np.ndarray, ground_truth_ids: np.ndarray, k: int) -> float:
    return float(np.mean([len(np.intersect1d(ids[i], ground_truth_ids[i])) / k for i in range(len(ids))]))


def main():
    # Data parameters
    d = 128  # Vector dimensionality
    n = 100  # Number of vectors in the database
    q = 20  # Number of query vectors
    k = 10  # Number of nearest neighbors to retrieve

    # HNSW parameters
    M = 8  # Number of neighbors
    ef_construction = 42  # Number of neighbors to consider for the construction of the graph
    ef_search = 12  # Search speed/quality trade-off parameter
    metric_faiss, metric_hnswlib = faiss.METRIC_L2, "l2"  # Metric to use for the distance calculation

    # Generate random data
    X = np.random.rand(n, d).astype("float32")  # Database of n=100 vectors of d=128 dimensions
    Q = np.random.rand(q, d).astype("float32")  # Set of q=20 query vectors of d=128 dimensions

    faiss_ids = faiss_index(X, d, M, metric_faiss, ef_construction, ef_search, Q, k)
    hnswlib_ids = hnswlib_index(X, d, M, metric_hnswlib, ef_construction, ef_search, Q, k)

    print(f"Do results of FAISS and hnswlib match? {np.all(faiss_ids == hnswlib_ids)}")

    # Calculate the ground truth -- brute force kNN search
    _, ground_truth_ids = faiss.knn(Q, X, k)

    # Calculate the recall
    recall_faiss = calculate_average_recall(faiss_ids, ground_truth_ids, k)
    recall_hnswlib = calculate_average_recall(hnswlib_ids, ground_truth_ids, k)

    print(f"Recall FAISS: {recall_faiss}")
    print(f"Recall hnswlib: {recall_hnswlib}")


if __name__ == "__main__":
    main()
