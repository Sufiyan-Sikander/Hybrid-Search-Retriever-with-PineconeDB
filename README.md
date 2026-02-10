# Hybrid-Search-Retriever-with-PineconeDB
This project builds a hybrid search retriever using Pinecone that combines semantic vector search with BM25-based lexical search to improve retrieval accuracy for RAG pipelines, chatbots, and document-based question answering




# Hybrid Search with LangChain and Pinecone

A demonstration of hybrid search capabilities combining dense vector embeddings and sparse BM25 keyword matching using LangChain and Pinecone.

## Overview

This project implements a hybrid search system that leverages both semantic and keyword-based search approaches:

- **Dense Vector Search**: Uses HuggingFace's `all-MiniLM-L6-v2` embeddings to capture semantic meaning
- **Sparse Keyword Search**: Employs BM25 encoding to match exact and partial keywords
- **Vector Database**: Pinecone serverless index for efficient storage and retrieval
- **Integration**: LangChain's PineconeHybridSearchRetriever for seamless orchestration

## Architecture

```
Text Input
    ↓
HuggingFace Embeddings (Dense) + BM25 Encoder (Sparse)
    ↓
Pinecone Hybrid Search Index
    ↓
PineconeHybridSearchRetriever
    ↓
Retrieved Results
```

## Key Components

1. **Embeddings**: HuggingFace all-MiniLM-L6-v2 model for dense vector representations
2. **Sparse Encoder**: BM25Encoder for keyword-based matching with TF-IDF scoring
3. **Vector Store**: Pinecone serverless index with dotproduct metric
4. **Retriever**: LangChain's PineconeHybridSearchRetriever combining both search methods

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
Create a `.env` file with:
```
PINECONE_API_KEY=<your-api-key>
HF_TOKEN=<your-huggingface-token>
```

3. Run the notebook:
```bash
jupyter notebook experiments.ipynb
```

## Usage

The notebook demonstrates:
- Initializing the Pinecone client
- Creating a serverless index
- Loading HuggingFace embeddings
- Training BM25 encoder on sample data
- Adding texts to the retriever
- Querying with natural language

## Example Query

```python
retriever.invoke("What city did i visit first")
```

## Dependencies

- `langchain`
- `langchain-community`
- `langchain-huggingface`
- `pinecone`
- `pinecone-text`
- `sentence-transformers`

## Files

- `experiments.ipynb`: Main notebook with implementation
- `bm25_values.json`: Persisted BM25 encoder state
- `requirements.txt`: Python dependencies
