# RAGAS-Evaluator
Repository Description: Llama Index and RAGAS Evaluator Welcome to the Llama Index and RAGAS Evaluator repository! This project showcases a comprehensive setup for building a powerful Retriever-Augmented Generation (RAG) system using the Llama Index library, FAISS for vector storage, and OpenAI embeddings for semantic parsing.

# Llama Index and RAGAS Evaluator

This repository demonstrates how to use Llama Index to create a powerful retriever augmented generation (RAG) app, utilizing FAISS for vector storage and OpenAI embeddings for semantic parsing. Additionally, it includes the RAGAS evaluation framework to measure the effectiveness of your models.

## Features

- **Web Data Loading**: Use BeautifulSoup to load data from web pages.
- **Vector Storage**: Utilize FAISS for efficient vector storage and retrieval.
- **OpenAI Embeddings**: Generate embeddings using OpenAI's models.
- **Semantic Parsing**: Implement semantic parsing with Llama Index.
- **RAGAS Evaluation**: Evaluate the RAG system using RAGAS metrics.

## Installation

1. **Clone the repository**
    ```bash
    git clone https://github.com/your-username/llama-index-ragas-evaluator.git
    cd llama-index-ragas-evaluator
    ```

2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up OpenAI API Key**
    ```python
    import os
    os.environ["OPENAI_API_KEY"] = 'your-openai-api-key'
    ```

## Usage

1. **Run the main script**
    ```python
    import nest_asyncio
    nest_asyncio.apply()

    import os
    import openai
    from getpass import getpass

    openai.api_key = getpass("Please provide your OpenAI Key: ")
    os.environ["OPENAI_API_KEY"] = openai.api_key

    from llama_index.readers.web import BeautifulSoupWebReader
    from llama_index.core import (
        load_index_from_storage,
        VectorStoreIndex,
        StorageContext,
    )

    loader = BeautifulSoupWebReader()
    documents = loader.load_data(urls=["https://en.wikipedia.org/wiki/2023_in_video_games"])
    index = VectorStoreIndex.from_documents(documents)

    # Further code follows...
    ```

2. **Evaluate with RAGAS**
    ```python
    from datasets import Dataset
    from ragas.metrics import context_precision, context_relevancy, answer_similarity
    from ragas import evaluate

    data_samples = {
        'question': [...],
        'answer': [...],
        'contexts': [...],
        'ground_truth': [...]
    }

    dataset = Dataset.from_dict(data_samples)
    score = evaluate(dataset, metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        context_relevancy
    ])
    score.to_pandas()
    ```

## Contributing

Feel free to open issues or submit pull requests with improvements.

## License

This project is licensed under the MIT License.
