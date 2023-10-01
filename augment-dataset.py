from embeddings.chroma_funcs import (
    get_closest_entries,
    generate_knowledge_base_from_hf_dataset,
    get_random_entries,
    get_embedding_model_name,
)
from datasets import load_dataset
from chromadb.utils import embedding_functions


def format_rag_examples(examples):
    if len(examples) == 1:
        return f"""
Given the following example:
### Input:
{examples[0]["question"]}

### Context:
{examples[0]["context"]}

### Response:
{examples[0]["answer"]}
"""
    formatted_examples = "\n".join(
        f"""Example {j+1}:
### Input:
{example["question"]}

### Context:
{example["context"]}

### Response:
{example["answer"]}
"""
        for j, example in enumerate(examples)
    )

    return f"""
Given the following examples:
{formatted_examples}"""


def get_examples(knowledge_base, data_point, n_examples, randomize=False):
    formatted_examples = ""
    if n_examples > 0:
        if randomize:
            formatted_examples = get_random_entries(knowledge_base, n_examples)[
                "metadatas"
            ]
        else:
            formatted_examples = get_closest_entries(
                knowledge_base,
                data_point["question"],
                "question",
                n_results=n_examples,
                db_id=data_point["db_id"],
            )["metadatas"][0]
        print(
            data_point["question"],
            " -> ",
            formatted_examples[0]["question"],
        )
        print(
            data_point["db_id"],
            " -> ",
            formatted_examples[0]["db_id"],
        )
        formatted_examples = format_rag_examples(formatted_examples)
    return formatted_examples


def generate_rag_sql_prompt(knowledge_base, data_point, n_examples, randomize=False):
    formatted_examples = get_examples(knowledge_base, data_point, n_examples, randomize)

    inference_prompt = f"""You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables. You must output the SQL query that answers the question.
{formatted_examples}
Please generate the SQL query that answers the following:
### Input:
{data_point["question"]}

### Context:
{data_point["context"]}

### Response:"""
    full_prompt = f"{inference_prompt}\n{data_point['answer']}"
    return full_prompt, inference_prompt


def add_prompt_features(example, knowledge_base, n_examples, randomize=False):
    # Add your logic to generate the extra feature here
    full_prompt, inference_prompt = generate_rag_sql_prompt(
        knowledge_base, example, n_examples, randomize
    )
    example["full_prompt"] = full_prompt
    example["inference_prompt"] = inference_prompt
    return example


def augment_dataset_with_prompts(
    dataset_name, knowledge_base, n_examples=1, randomize=False
):
    dataset_dict = load_dataset(dataset_name)

    for split, dataset in dataset_dict.items():
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
dataset_name = "samlhuillier/sql-create-context-spider-intersect"
knowledge_base = generate_knowledge_base_from_hf_dataset(
    dataset_name, "question", openai_ef
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
