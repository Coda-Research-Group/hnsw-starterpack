# HNSW Starterpack

There are two main implementations of HNSW:
- `hnswlib` – https://github.com/nmslib/hnswlib
- `faiss` – https://github.com/facebookresearch/faiss

This repository contains a simple example of how to use both of them.

## Setup the environment

```shell
conda env create --file environment.yml
conda activate hnsw-starterpack
```

## Run the example

The example creates a random set of database vectors `X` and a random set of queries `Q`.
It then constructs an HNSW index using `faiss` and `hnswlib` and searches for `k` nearest neighbors.
Finally, it calculates the average recall of the two implementations.

```shell
python main.py
```

## Other resources

- [HNSW algorithm parameters (hnswlib)](https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md)
- [HNSW algorithm – Pinecone (FAISS)](https://www.pinecone.io/learn/series/faiss/hnsw/)
