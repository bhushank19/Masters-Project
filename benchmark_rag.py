from rag_integration import setup_rag_with_web_data
from web_scraping import fetch_product_data
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_cohere import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain.schema import Document
import asyncio
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Fetch the Cohere API key from environment variables
cohere_api_key = os.getenv("COHERE_API_KEY")
if not cohere_api_key:
    raise ValueError("COHERE_API_KEY environment variable not set")

# Metric Calculation Functions
def calculate_precision(retrieved_docs, relevant_docs):
    relevant_retrieved = [doc for doc in retrieved_docs if doc in relevant_docs]
    return len(relevant_retrieved) / len(retrieved_docs) if retrieved_docs else 0

def calculate_recall(retrieved_docs, relevant_docs):
    relevant_retrieved = [doc for doc in retrieved_docs if doc in relevant_docs]
    return len(relevant_retrieved) / len(relevant_docs) if relevant_docs else 0

def calculate_relevancy(retrieved_docs, relevant_docs):
    return len(set(retrieved_docs) & set(relevant_docs)) / len(set(retrieved_docs + relevant_docs)) if retrieved_docs else 0

# Benchmark Retrieval Components
async def benchmark_retrieval_components(user_query, top_n=5):
    # Fetch product data from Amazon
    products = await fetch_product_data(user_query)

    if not products:
        print("No products found for the query.")
        return

    # Define dynamic ground truth as the top N products by rating
    ground_truth_relevant_docs = [p['title'] for p in sorted(products, key=lambda x: -float(x['rating']))[:top_n]]

    # 1. BM25-only retrieval
    bm25_retriever = BM25Retriever.from_documents([Document(page_content=p['title']) for p in products])
    bm25_results = bm25_retriever.get_relevant_documents(user_query)
    bm25_titles = [doc.page_content for doc in bm25_results]

    # Calculate metrics for BM25
    precision_bm25 = calculate_precision(bm25_titles, ground_truth_relevant_docs)
    recall_bm25 = calculate_recall(bm25_titles, ground_truth_relevant_docs)
    relevancy_bm25 = calculate_relevancy(bm25_titles, ground_truth_relevant_docs)
    
    print(f"BM25 - Precision: {precision_bm25:.2f}, Recall: {recall_bm25:.2f}, Relevancy: {relevancy_bm25:.2f}")

    # 2. Ensemble retrieval (BM25 + FAISS with weights)
    vectorstore = await setup_rag_with_web_data(products)
    keyword_retriever = BM25Retriever.from_documents([Document(page_content=p['title']) for p in products])
    ensemble_retriever = EnsembleRetriever(retrievers=[vectorstore.base_retriever, keyword_retriever], weights=[0.5, 0.5])
    ensemble_results = ensemble_retriever.get_relevant_documents(user_query)
    ensemble_titles = [doc.page_content for doc in ensemble_results]

    # Calculate metrics for Ensemble
    precision_ensemble = calculate_precision(ensemble_titles, ground_truth_relevant_docs)
    recall_ensemble = calculate_recall(ensemble_titles, ground_truth_relevant_docs)
    relevancy_ensemble = calculate_relevancy(ensemble_titles, ground_truth_relevant_docs)
    
    print(f"Ensemble - Precision: {precision_ensemble:.2f}, Recall: {recall_ensemble:.2f}, Relevancy: {relevancy_ensemble:.2f}")

    # 3. Re-ranking with Cohere
    compressor = CohereRerank(model="rerank-english-v2.0", cohere_api_key=cohere_api_key)
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble_retriever)
    reranked_results = compression_retriever.get_relevant_documents(user_query)
    reranked_titles = [doc.page_content for doc in reranked_results]

    # Calculate metrics for Cohere re-ranking
    precision_reranked = calculate_precision(reranked_titles, ground_truth_relevant_docs)
    recall_reranked = calculate_recall(reranked_titles, ground_truth_relevant_docs)
    relevancy_reranked = calculate_relevancy(reranked_titles, ground_truth_relevant_docs)
    
    print(f"Reranked - Precision: {precision_reranked:.2f}, Recall: {recall_reranked:.2f}, Relevancy: {relevancy_reranked:.2f}")


# Example usage
if __name__ == "__main__":
    user_query = "Logitech mouse"
    
    # Run benchmarks with dynamic ground truth generated from the scraped data
    asyncio.run(benchmark_retrieval_components(user_query, top_n=5))
