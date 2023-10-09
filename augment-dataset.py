from embeddings.chroma_funcs import (
    get_closest_entries,
    generate_knowledge_base_from_hf_dataset,
    get_random_entries,
    get_embedding_model_name,
)
from datasets import load_dataset
from chromadb.utils import embedding_functions
from prompt_setup import (
    generate_generic_prompt,
)


def format_math_example(example):
    inference_prompt = f"""### Problem:
{example["question"]}

### Answer:"""
    full_prompt = f"{inference_prompt}\n{example['answer']}"
    return full_prompt, inference_prompt


prompt = "Solve the following math problem thinking step-by-step:"


def add_prompt_features(example, knowledge_base, embed_feature, n_examples):
    # Add your logic to generate the extra feature here
    full_prompt, inference_prompt = generate_generic_prompt(
        knowledge_base, example, embed_feature, n_examples, prompt, format_math_example
    )
    print("inference_prompt", inference_prompt)
    print("full_prompt", full_prompt)
    example["full_prompt"] = full_prompt
    example["inference_prompt"] = inference_prompt
    return example


def augment_dataset_with_prompts(
    dataset_args, knowledge_base, embed_feature, n_examples=1
):
    # print("dataset_args", **dataset_args)
    dataset_dict = load_dataset(*dataset_args.values())

    for split, dataset in dataset_dict.items():
        print(dataset)
        dataset = dataset.map(
            lambda example: add_prompt_features(
                example, knowledge_base, embed_feature, n_examples=n_examples
            ),
        )

        # TODO: add in embedding function:
        embedding_function = get_embedding_model_name(
            knowledge_base._embedding_function
        )
        # emb_fn_string = ""
        dataset_name = dataset_args["dataset_name"]
        filename = f"{dataset_name.replace('/', '-')}-{split}-with-{n_examples}-examples-{embedding_function}.jsonl"

        # Save the dataset as a JSON file
        dataset.to_json(filename)


openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    # api_key="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",
    model_name="text-embedding-ada-002",
)

default_ef = embedding_functions.DefaultEmbeddingFunction()

print(default_ef.model)

# %%
# so first we need to generate the knowledge_base
# dataset_name = "gsm8k"
embedding_feature = "question"
dataset_parameters = {"dataset_name": "gsm8k", "config_name": "main"}

knowledge_base = generate_knowledge_base_from_hf_dataset(
    dataset_parameters, embedding_feature, openai_ef
)

augment_dataset_with_prompts(
    dataset_parameters, knowledge_base, embedding_feature, n_examples=1
)
