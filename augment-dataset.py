from embeddings.chroma_funcs import (
    get_closest_entries,
    generate_knowledge_base_from_hf_dataset,
    get_random_entries,
    get_embedding_model_name,
)
from datasets import load_dataset
from chromadb.utils import embedding_functions
from prompt_setup import (
    # format_rag_sql_examples,
    # get_sql_examples,
    generate_rag_sql_prompt,
    generate_rag_func_representation_prompt,
    generate_gsm8k_prompt,
)


def add_prompt_features(example, knowledge_base, n_examples, randomize=False):
    # Add your logic to generate the extra feature here
    full_prompt, inference_prompt = generate_gsm8k_prompt(
        knowledge_base, example, n_examples, randomize
    )
    print("full_prompt", full_prompt)
    example["full_prompt"] = full_prompt
    example["inference_prompt"] = inference_prompt
    return example


def augment_dataset_with_prompts(
    dataset_name, knowledge_base, n_examples=1, randomize=False
):
    dataset_dict = load_dataset(dataset_name, "main")

    for split, dataset in dataset_dict.items():
        print(dataset)
        dataset = dataset.map(
            lambda example: add_prompt_features(
                example, knowledge_base, n_examples=n_examples, randomize=randomize
            ),
        )

        # TODO: add in embedding function:
        embedding_function = get_embedding_model_name(
            knowledge_base._embedding_function
        )
        emb_fn_string = ""
        if not randomize:
            emb_fn_string = f"-emb_fn-{embedding_function}"

        filename = f"{dataset_name.replace('/', '-')}-{split}-with-{n_examples}-examples-random-{randomize}{emb_fn_string}.jsonl"

        # Save the dataset as a JSON file
        dataset.to_json(filename)


openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="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",
    model_name="text-embedding-ada-002",
)

# print(openai_ef._model_name)
# abc = openai_ef(
#     [
#         "hell owrld",
#     ]
# )

default_ef = embedding_functions.DefaultEmbeddingFunction()

print(default_ef.model)

# %%
# so first we need to generate the knowledge_base
dataset_name = "gsm8k"
embedding_feature = "question"
knowledge_base = generate_knowledge_base_from_hf_dataset(
    dataset_name, embedding_feature, openai_ef
)
print(knowledge_base.count())
print(get_embedding_model_name(knowledge_base._embedding_function))
# entries = get_random_entries(knowledge_base, 1)
# print(entries)
augment_dataset_with_prompts(
    dataset_name, knowledge_base, n_examples=1, randomize=False
)


# %%
# test_datapoint = {
#     "question": "What is the average horsepower for all cars produced before 1980 ?",
#     "context": "CREATE TABLE cars_data (horsepower INTEGER, year INTEGER)",
#     "answer": "select avg(horsepower) from cars_data where year  <  1980;",
#     "db_id": "car_1",
# }

# full_prompt, inference_prompt = generate_rag_sql_prompt(
#     knowledge_base, test_datapoint, n_examples=2
# )
# print(full_prompt)
