import chromadb
from chromadb.utils import embedding_functions

from datasets import load_dataset

client = chromadb.PersistentClient(path="./chroma_store")

# collection = client.get_or_create_collection(name="dataset")
# print(collection.count())

# so what functions do we want

# given a dataset object, the feature we want to embed.
# maybe we just want a lookup no...Like let me get closest match given some string. Or maybe even take in the path to a given dataset and the feature to embed and total no of values to return


def setup_collection(hf_dataset_name, feature_to_embed, dataset_split, collection):
    # 1. load dataset object
    # 2. embed each feature property
    # 3.
    dataset = load_dataset(hf_dataset_name, split=dataset_split)
    # print(dataset["question"])
    # print(type(dataset[feature_to_embed]))
    dict = dataset.to_dict()
    print(type(dict))
    # print(dataset.to_dict()[0])
    collection.add(
        documents=dataset[feature_to_embed],
        metadatas=[item for item in dataset],
        ids=[f"id{i+1}" for i in range(len(dataset))],
    )
    return collection


# TODO: this dataset_path could be both a huggingface name or a local path. Then we leverage transformers to pull it agnostically
def get_closest_entries(
    hf_dataset_name, embed_feature, query, n_results=5, dataset_split="train"
):
    collection_name = hf_dataset_name.replace("/", "-") + "-" + dataset_split
    # client.delete_collection(name=collection_name)
    collection = client.get_or_create_collection(name=collection_name)
    print("collection.count()", collection.count())
    if collection.count() == 0:
        # embed and create collection...
        print("hey x")
        collection = setup_collection(
            hf_dataset_name, embed_feature, dataset_split, collection
        )

    print("collection.count()", collection.count())

    results = collection.query(
        query_texts=[query],  # TODO: look into what multiple queries will do here.
        n_results=n_results,
        where={embed_feature: {"$ne": query}},
    )
    print(results["metadatas"])
    # TODO: add clause to remove exact matches
    return results


# get_closest_entries(
#     "samlhuillier/sql-create-context-spider-intersect",
#     "question",
#     "get the best football players",
#     n_results=3,
#     dataset_split="train",
# )
