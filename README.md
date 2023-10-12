<a name="readme-top"></a>


<!-- PROJECT LOGO -->
<br />
<div align="center">


  <h3 align="center">Tune the Rag</h3>

  <p align="center">
    Convert any Huggingface dataset to a retrieval-augmented dataset
    <br />
    <a href="https://ragntune.com/blog/Fine-tuning-an-LLM-to-be-good-at-RAG"><strong>Read the blog post »</strong></a>
   
  </p>
</div>





<!-- ABOUT THE PROJECT -->
## About The Project
- Use this repo to augment prompts with semantically similar prompts from the training set.
- Choose the embedding model to perform the semantic search.
- Customise the prompts.

For example, the data point from an SQL dataset ```{"question": "How many singers do we have?", "context": "CREATE TABLE singer (Id VARCHAR)"}``` gets prompted to be:
```
You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables. You must output the SQL query that answers the question.

Given the following example:
### Input:
How many artists do we have?

### Context:
CREATE TABLE artist (Id VARCHAR)

### Response:
SELECT count(*) FROM artist

Please generate the SQL query that answers the following:
### Input:
How many singers do we have?

### Context:
CREATE TABLE singer (Id VARCHAR)

### Response:
```
And these are the results of using retrieval-augmented prompts vs few-shot prompts:

![Description or Alt Text](https://ragntune.com/static/images/tuneforrag/chart.svg)


<!-- GETTING STARTED -->
## Getting Started


### Prerequisites

1. Clone repo
```sh
git clone https://github.com/samlhuillier/tunetherag.git
```
2. Install requirements
  ```sh
  pip install -r requirements.txt
  ```

### Creating a RAG dataset

1. Open ```tunetherag.ipynb```
2. In the third cell, modify the dataset loading arguments:
   ```python
   embedding_feature = "question"
   dataset_parameters = {"dataset_name": "gsm8k", "config_name": "main"}
   ```
   (```embedding_feature``` is what will be embedded and ```dataset_parameters``` will be plugged into ```load_dataset```)
3. Setup prompts:
   ```python
   def format_math_example(example):
    inference_prompt = f"""### Problem:
    {example["question"]}

    ### Answer:"""

    full_prompt = f"{inference_prompt}\n{example['answer']}"

    return full_prompt, inference_prompt
   
   math_prompt = "Solve the following math problem thinking step-by-step:"
   ```
4. Run cells to generate the retrieval augmented dataset which will be saved as a set of json files per split!

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License.







