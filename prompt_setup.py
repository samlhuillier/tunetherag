from embeddings.chroma_funcs import (
    get_closest_entries,
    get_random_entries,
)


def format_rag_sql_examples(examples):
    def format_example(i, example):
        return f"""Example {i+1}:
### Input:
{example["question"]}

### Context:
{example["context"]}

### Response:
{example["answer"]}
"""

    formatted_examples = "\n".join(
        format_example(i, example) for i, example in enumerate(examples)
    )

    prefix = (
        "Given the following example:"
        if len(examples) == 1
        else "Given the following examples:"
    )

    return f"""
{prefix}
{formatted_examples}"""


def get_examples_from_db(
    knowledge_base, data_point, n_examples, embed_feature, randomize=False
):
    examples = []
    if n_examples > 0:
        if randomize:
            examples = get_random_entries(knowledge_base, n_examples)["metadatas"]
        else:
            examples = get_closest_entries(
                knowledge_base,
                data_point[embed_feature],
                embed_feature,
                n_results=n_examples,
            )["metadatas"][0]
        print(data_point[embed_feature], " -> ", examples[0][embed_feature])
    return examples


def generate_rag_sql_prompt(knowledge_base, data_point, n_examples, randomize=False):
    examples = get_examples_from_db(
        knowledge_base, data_point, n_examples, "question", randomize
    )
    if len(examples) > 0:
        formatted_examples = format_rag_sql_examples(examples)
    else:
        formatted_examples = ""
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


def format_rag_func_rep_examples(examples):
    def format_example(i, example):
        return f"""Example {i+1}:
### Target sentence:
{example["target"]}

### Meaning representation:
{example["meaning_representation"]}
"""

    formatted_examples = "\n".join(
        format_example(i, example) for i, example in enumerate(examples)
    )

    prefix = (
        "Given the following example:"
        if len(examples) == 1
        else "Given the following examples:"
    )

    return f"""
{prefix}
{formatted_examples}"""


def format_rag_gsm8k_examples(examples):
    def format_example(i, example):
        return f"""Problem:
{example["question"]}

Answer:
{example["answer"]}
"""

    formatted_examples = "\n".join(
        format_example(i, example) for i, example in enumerate(examples)
    )

    prefix = (
        "Given the following example:"
        if len(examples) == 1
        else "Given the following examples:"
    )

    return f"""
{prefix}
{formatted_examples}"""


def generate_rag_func_representation_prompt(
    knowledge_base, data_point, n_examples, randomize=False
):
    examples = get_examples_from_db(
        knowledge_base, data_point, n_examples, "target", randomize
    )
    if len(examples) > 0:
        formatted_examples = format_rag_func_rep_examples(examples)
    else:
        formatted_examples = ""

    inference_prompt = f"""Given a target sentence construct the underlying meaning representation of the input sentence as a single function with attributes and attribute values.
This function should describe the target string accurately and the function must be one of the following ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation', 'recommend', 'request_attribute'].
The attributes must be one of the following: ['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating', 'genres', 'player_perspective', 'has_multiplayer', 'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']
{formatted_examples}
Please generate the underlying meaning representation of the following:
### Target sentence:
{data_point["target"]}

### Meaning representation:"""

    full_prompt = f"{inference_prompt}\n{data_point['meaning_representation']}"
    return full_prompt, inference_prompt


def generate_gsm8k_prompt(knowledge_base, data_point, n_examples, randomize=False):
    examples = get_examples_from_db(
        knowledge_base, data_point, n_examples, "question", randomize
    )
    if len(examples) > 0:
        formatted_examples = format_rag_gsm8k_examples(examples)
    else:
        formatted_examples = ""

    inference_prompt = f"""{formatted_examples}Solve the following math problem thinking step-by-step:
Problem:
{data_point["question"]}

Answer:"""

    full_prompt = f"{inference_prompt}\n{data_point['answer']}"
    return full_prompt, inference_prompt
