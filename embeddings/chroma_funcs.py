import chromadb
from chromadb.utils import embedding_functions

from datasets import load_dataset
import random

client = chromadb.PersistentClient(path="./chroma_store")


def index_to_id(index):
    return f"id{index+1}"


def generate_random_numbers(n):
    return [random.random() for _ in range(n)]


def dataset_chunks(dataset, n):
    """Yield successive n-sized chunks from the dataset."""
    for i in range(0, len(dataset), n):
        start_idx = i
        end_idx = min(i + n, len(dataset))
        yield dataset.select(range(start_idx, end_idx))


def get_embedding_model_name(emb_fn):
    model_name = "default-emb-fn"
    if hasattr(emb_fn, "_model_name"):
        model_name = emb_fn._model_name
    return model_name


def generate_knowledge_base_from_hf_dataset(
    hf_dataset_name, embed_feature, emb_fn, dataset_split="train"
):
    # so here, we should set the name of the collection based on the emb_fn
    print("embd_fn", emb_fn)
    model_name = get_embedding_model_name(emb_fn)
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

        # Initialize a global counter for unique IDs
        global_counter = 0
        chunks = dataset_chunks(dataset, chunk_size)
        for chunk in chunks:
            metadatas = [item for item in chunk]
            documents = chunk[embed_feature]

            # Generate unique IDs based on the global counter
            ids = [f"id{global_counter + i + 1}" for i in range(len(chunk))]

            # Update the global counter
            global_counter += len(chunk)

            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )
        # collection.add(
        #     documents=dataset[embed_feature],
        #     metadatas=[item for item in dataset],
        #     ids=[f"id{i+1}" for i in range(len(dataset))],
        # )
    print("Collection.count is: ", collection.count())
    return collection


def get_closest_entries(
    collection, query, embed_feature, n_results=5, db_id="sdojfaosdijfaposdfjia"
):
    results = collection.query(
        query_texts=[query],  # TODO: look into what multiple queries will do here.
        n_results=n_results,
        # where={embed_feature: {"$ne": query}},  # exact match is train/val contamination
        where={
            "$and": [
                {embed_feature: {"$ne": query}},
                {"db_id": {"$ne": db_id}},
            ]
        }
        # TODO: test this:
        # where_document={"$ne": query},  # exact match is train/val contamination
    )
    return results


def get_random_entries(collection, n_results=1):
    # so we just query the length of the collection then work
    # total_items = collection.count()
    print(collection.count())
    indexes = random.sample(range(collection.count()), n_results)
    ids = [index_to_id(index) for index in indexes]
    # full_collection =
    return collection.get(ids=ids)
    results = collection.query(
        query_texts=[""],  # TODO: look into what multiple queries will do here.
        n_results=n_results,
        # where={embed_feature: {"$ne": query}},  # exact match is train/val contamination
        # TODO: test this:
        # where_document={"$ne": query},  # exact match is train/val contamination
    )
    return results
