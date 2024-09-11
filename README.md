
# Advanced RAG System for Real-Time Recommendations

## Overview

This project, titled **"Advanced RAG System for Real-Time Recommendations: Integrating Hybrid Search and Re-Ranking with Dynamic Data"**, focuses on developing a state-of-the-art product recommendation system that leverages real-time web scraping and cutting-edge AI techniques such as **Large Language Models (LLMs)**, **Retrieval-Augmented Generation (RAG)**, and **Hybrid Search**.

### Thesis Context
This project was developed as part of the MSc in Computer Science (Artificial Intelligence) at the University of Galway, under the supervision of **Dr. Effirul Ramlan**. The core objective is to integrate real-time web scraping with RAG and AI-driven models to generate accurate and dynamic product recommendations.

**Key Features:**
- **Real-Time Data Retrieval**: Web scraping using Playwright and BeautifulSoup for live product data from Amazon.
- **Hybrid Search**: Combines both keyword-based (BM25) and semantic search (FAISS) techniques for improved accuracy.
- **RAG (Retrieval-Augmented Generation)**: Using HuggingFace embeddings, vector databases, and Cohere Rerank for re-ranking and contextual compression.
- **LLM Integration**: Query refinement and product recommendation generation using **Mistral 7B** LLM via Ollama.

## Technologies Used

- **Web Scraping**: Playwright, BeautifulSoup, DiskCache (for caching scraped results).
- **LLMs**: Mistral 7B model via **Ollama** for query refinement and response generation.
- **Hybrid Search**: Combining BM25 for keyword retrieval and FAISS for semantic search.
- **Vector Database**: FAISS for storing and retrieving document embeddings.
- **Reranking and Compression**: Cohere Re-ranker for improving the quality of recommendations.
- **Frameworks**: LangChain for workflow orchestration, Streamlit for a user-friendly interface.
- **Caching**: DiskCache for efficient retrieval of web scraping results.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Architecture](#architecture)
- [Performance Benchmarking](#performance-benchmarking)
- [Screenshots](#screenshots)
- [Future Work](#future-work)
- [Acknowledgements](#acknowledgements)

## Installation

### Prerequisites
Ensure that the following are installed:
- Python 3.8 or higher
- Docker (optional, for web scraping)

### Steps
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/rag-product-recommendation.git
   cd rag-product-recommendation
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the root directory:
   ```
   COHERE_API_KEY=<your_cohere_api_key>
   ```

4. **Install Playwright** (for web scraping):
   ```bash
   playwright install
   ```

### Docker Setup (Optional)
You can use Docker to run the web scraping component in an isolated environment:
```bash
docker build -t product-recommendation .
docker run -p 8501:8501 product-recommendation
```

## Usage

### Streamlit Frontend
To launch the Streamlit frontend for product search and recommendation:
```bash
streamlit run app.py
```
The Streamlit app allows users to input queries and get AI-powered product recommendations.

### Backend
For backend operations, you can use the individual modules:
- **Web Scraping**:
   ```python
   from web_scraping import fetch_product_data
   products = asyncio.run(fetch_product_data("Logitech Mouse"))
   ```

- **Recommendation Workflow**:
   ```python
   from rag_integration import rag_workflow
   recommendations = asyncio.run(rag_workflow("Logitech Mouse", products))
   ```

## Architecture

This system is built around **Retrieval-Augmented Generation (RAG)**, integrating both web scraping for real-time data and advanced AI models for query refinement and recommendation generation. Below is a breakdown of the architecture:

1. **Real-Time Data Collection**: 
   - Utilizes Playwright and BeautifulSoup for scraping live product data from Amazon.
   - Data such as titles, descriptions, prices, and reviews are extracted, processed, and cached.
   
2. **Query Refinement with LLM**:
   - The user's input query is refined using **Mistral 7B** LLM, making it more suitable for e-commerce searches.
   
3. **Document Embedding**:
   - The product data is split into chunks and embedded using **Sentence Transformers** to create dense vectors.
   
4. **Hybrid Search**:
   - Combines **BM25** (keyword-based) and **FAISS** (semantic-based) retrieval methods to retrieve relevant product data.
   
5. **Cohere Re-ranking**:
   - The retrieved data is re-ranked using Cohere’s re-ranking API to ensure the most relevant results are presented.
   
6. **Product Recommendation Generation**:
   - The refined data is used to generate product recommendations, ranked by rating and reviews, and presented to the user.

## Performance Benchmarking

The system has been evaluated across multiple metrics:
- **Web Scraping**: Benchmarked for speed and accuracy (e.g., 24 products scraped in 0.18 seconds for "Samsung Galaxy S23").
- **Query Refinement**: LLM refinement time measured (e.g., 12.84 seconds for "Samsung Galaxy").
- **RAG Pipeline**: Precision, recall, and relevancy scores for retrieval and re-ranking stages (e.g., BM25 precision = 0.75).

## Screenshots

### Example Query: "Logitech Mouse"
- **Initial Input**: "Suggest a good wireless mouse from Logitech."
- **Refined Query**: "wireless+mouse+logitech"
- **Top Recommendations**:
  1. Logitech MX Master 3
  2. Logitech G502 Wireless
  
(Screenshot here)

## Future Work

- **Dynamic Data Sources**: Extend scraping to more e-commerce platforms beyond Amazon.
- **Real-Time Monitoring**: Implement real-time error handling for web scraping.
- **Improved Query Refinement**: Use fine-tuned LLMs for better performance across diverse query types.


