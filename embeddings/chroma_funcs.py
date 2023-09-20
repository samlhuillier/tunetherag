import chromadb
from chromadb.utils import embedding_functions

from datasets import load_dataset

client = chromadb.PersistentClient(path="./chroma_store")


def setup_collection(hf_dataset_name, feature_to_embed, dataset_split, collection):
    dataset = load_dataset(hf_dataset_name, split=dataset_split)
    collection.add(
        documents=dataset[feature_to_embed],
        metadatas=[item for item in dataset],
        ids=[f"id{i+1}" for i in range(len(dataset))],
    )
    return collection


def get_closest_entries(
    hf_dataset_name, embed_feature, query, n_results=5, dataset_split="train"
):
    collection_name = hf_dataset_name.replace("/", "-") + "-" + dataset_split
    collection = client.get_or_create_collection(name=collection_name)

    if collection.count() == 0:
        collection = setup_collection(
            hf_dataset_name, embed_feature, dataset_split, collection
        )

    results = collection.query(
        query_texts=[query],  # TODO: look into what multiple queries will do here.
        n_results=n_results,
        where={embed_feature: {"$ne": query}},
    )
    print(results["metadatas"])
    return results


# get_closest_entries(
#     "samlhuillier/sql-create-context-spider-intersect",
#     "question",
#     "get the best football players",
#     n_results=3,
#     dataset_split="train",
# )
