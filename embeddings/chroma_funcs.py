import chromadb
from chromadb.utils import embedding_functions

from datasets import load_dataset

client = chromadb.PersistentClient(path="./chroma_store")


# def setup_collection(hf_dataset_name, feature_to_embed, dataset_split, collection):
#     dataset = load_dataset(hf_dataset_name, split=dataset_split)
#     collection.add(
#         documents=dataset[feature_to_embed],
#         metadatas=[item for item in dataset],
#         ids=[f"id{i+1}" for i in range(len(dataset))],
#     )
#     return collection


def dataset_chunks(dataset, n):
    """Yield successive n-sized chunks from the dataset."""
    for i in range(0, len(dataset), n):
        start_idx = i
        end_idx = min(i + n, len(dataset))
        yield dataset.select(range(start_idx, end_idx))


def generate_knowledge_base_from_hf_dataset(
    hf_dataset_name, embed_feature, emb_fn, dataset_split="train"
):
    # so here, we should set the name of the collection based on the emb_fn
    print("embd_fn", emb_fn)
    model_name = "default-emb-fn"
    if hasattr(emb_fn, "_model_name"):
        model_name = emb_fn._model_name
    # emb_fn_name = emb_fn._model_name
    collection_name = (
        hf_dataset_name.replace("/", "-") + "-" + dataset_split + "-" + model_name
    )[:62]

    # client.delete_collection(collection_name)
    collection = client.get_or_create_collection(
        name=collection_name, embedding_function=emb_fn
    )

    if collection.count() == 0:
        dataset = load_dataset(hf_dataset_name, split=dataset_split)
        chunk_size = 300
        for chunk in dataset_chunks(dataset, chunk_size):
            collection.add(
                documents=chunk[embed_feature],
                metadatas=[item for item in chunk],
                ids=[f"id{i+1}" for i in range(len(chunk))],
            )
        # collection.add(
        #     documents=dataset[embed_feature],
        #     metadatas=[item for item in dataset],
        #     ids=[f"id{i+1}" for i in range(len(dataset))],
        # )
    return collection


def get_closest_entries(collection, query, embed_feature, n_results=5):
    results = collection.query(
        query_texts=[query],  # TODO: look into what multiple queries will do here.
        n_results=n_results,
        where={embed_feature: {"$ne": query}},  # exact match is train/val contamination
        # TODO: test this:
        # where_document={"$ne": query},  # exact match is train/val contamination
    )
    return results


# def old_get_closest_entries(
#     hf_dataset_name, embed_feature, query, n_results=5, dataset_split="train"
# ):
#     collection_name = hf_dataset_name.replace("/", "-") + "-" + dataset_split
#     collection = client.get_or_create_collection(name=collection_name)

#     if collection.count() == 0:
#         collection = setup_collection(
#             hf_dataset_name, embed_feature, dataset_split, collection
#         )

#     results = collection.query(
#         query_texts=[query],  # TODO: look into what multiple queries will do here.
#         n_results=n_results,
#         where={embed_feature: {"$ne": query}},  # exact match is train/val contamination
#     )
#     # TODO: throw error here if there are any exact matches
#     return results


# get_closest_entries(
#     "samlhuillier/sql-create-context-spider-intersect",
#     "question",
#     "get the best football players",
#     n_results=3,
#     dataset_split="train",
# )
