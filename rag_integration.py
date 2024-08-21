from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.llms import Ollama
from langchain.retrievers import EnsembleRetriever, BM25Retriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from concurrent.futures import ThreadPoolExecutor
import torch
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Fetch the Cohere API key from environment variables
cohere_api_key = os.getenv("COHERE_API_KEY")
if not cohere_api_key:
    raise ValueError("COHERE_API_KEY environment variable not set")

# Check if MPS (Apple Silicon) is available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load the Ollama LLaMA model only once and reuse it across all functions
llm = Ollama(model="mistral:latest")  # Use a smaller model for faster performance

async def refine_query_with_llm(user_query):
    # Create a concise prompt for the LLM, focusing on generating simple and short search keywords
    refinement_prompt = (
        f"Transform the following query into a concise, keyword-based format suitable for an e-commerce search engine. "
        f"Include only essential keywords related to the main product and its key features, don't include key feature word. Remove any extraneous details. "
        f"Ensure the result is a simple, short query without unnecessary descriptions: '{user_query}'."
    )
    
    # Get response from LLM
    refined_query_response = llm.generate(prompts=[refinement_prompt])

    # Extract the refined query and post-process it
    refined_query = refined_query_response.generations[0][0].text.strip()

    # Ensure the query is in a simple, concise format by converting spaces to plus signs for URL compatibility
    refined_query = refined_query.lower().replace(" ", "+")
    
    # Limit the number of keywords to a maximum of 10 to keep it concise and focused
    keywords = refined_query.split("+")
    max_keywords = 10
    if len(keywords) > max_keywords:
        refined_query = "+".join(keywords[:max_keywords])
    
    return refined_query


async def filter_irrelevant_products(products, refined_query):
    filtered_products = []
    
    for product in products:
        product_info = f"Title: {product['title']}\nDescription: {product['description']}\n"
        filter_prompt = (
            f"Given the refined query '{refined_query}', determine if the following product is relevant. "
            f"Respond with 'Yes' if it is relevant, and 'No' if it is irrelevant.\n\n{product_info}"
        )
        
        filter_response = llm.generate(prompts=[filter_prompt])
        is_relevant = filter_response.generations[0][0].text.strip().lower()
        
        if is_relevant == 'yes':
            filtered_products.append(product)
    
    # Log the number of relevant products after filtering
    print(f"Number of relevant products after filtering: {len(filtered_products)}")
    for product in filtered_products[:5]:  # Print first 5 filtered products for inspection
        print(product)
    
    return filtered_products


# 1. Setup RAG with Web Data
async def setup_rag_with_web_data(products):
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {"device": device}
    embeddings_model = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs=model_kwargs)
    
    product_docs = []

    def process_product(product):
        product_text = (
            f"Title: {product['title']}\n"
            f"Description: {product['description']}\n"  # Ensure description is included
            f"Price: {product['price']}\n"
            f"Rating: {product['rating']}\n"
            f"Review Count: {product['review_count']}\n"
            f"Link: {product['link']}"  # Include the product link
        )
        return product_text

    with ThreadPoolExecutor() as executor:
        product_texts = list(executor.map(process_product, products))
        
        for product_text in product_texts:
            product_docs.append(Document(page_content=product_text))
    
    # Splitting the data into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50, separators=["\n\n", "\n", " "])
    docs = text_splitter.split_documents(product_docs)

    # Create embeddings
    doc_contents = [doc.page_content for doc in docs]
    with ThreadPoolExecutor() as executor:
        embeddings = list(executor.map(embeddings_model.embed_documents, doc_contents))
    
    # Load embeddings into FAISS
    vectorstore = FAISS.from_documents(docs, embeddings_model)

    # Set up BM25Retriever
    keyword_retriever = BM25Retriever.from_documents(docs)

    # Ensemble retriever
    ensemble_retriever = EnsembleRetriever(retrievers=[vectorstore.as_retriever(), keyword_retriever], weights=[0.5, 0.5])

    # Re-ranking with Cohere
    compressor = CohereRerank(cohere_api_key=cohere_api_key)
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble_retriever)

    return compression_retriever



# 2. RAG Workflow using the existing `generate_recommendations_with_llm`
async def rag_workflow(refined_query, products):
    print(f"Number of products passed to RAG: {len(products)}")
    
    retriever = await setup_rag_with_web_data(products)
    retrieved_docs = retriever.get_relevant_documents(refined_query)
    
    relevant_products = []
    for doc in retrieved_docs:
        product_details = {}
        lines = doc.page_content.split("\n")
        for line in lines:
            if line.startswith("Title:"):
                product_details["title"] = line.replace("Title:", "").strip()
            elif line.startswith("Description:"):
                product_details["description"] = line.replace("Description:", "").strip() or "No description available"
            elif line.startswith("Price:"):
                product_details["price"] = line.replace("Price:", "").strip()
            elif line.startswith("Rating:"):
                product_details["rating"] = line.replace("Rating:", "").strip()
            elif line.startswith("Review Count:"):
                product_details["review_count"] = int(line.replace("Review Count:", "").strip())
            elif line.startswith("Link:"):
                product_details["link"] = line.replace("Link:", "").strip()

        if "link" in product_details and "description" in product_details:
            relevant_products.append(product_details)
    
    print(f"Number of relevant products after RAG retrieval: {len(relevant_products)}")
    for product in relevant_products[:5]:
        print(product)
    
    recommendations = await generate_recommendations_with_llm(relevant_products)
    return recommendations



async def generate_recommendations_with_llm(products):
    # # Filter out products without a link
    # filtered_products = [product for product in filtered_products if 'link' in product]

    # if not filtered_products:
    #     return "No products with valid links found."

    # # Further safeguard against malformed products
    # filtered_products = [
    #     product for product in filtered_products 
    #     if all(key in product for key in ['title', 'price', 'rating', 'review_count', 'link'])
    # ]

    # if not filtered_products:
    #     return "No products with valid links found."

    # Rank products by rating and review count
    sorted_products = sorted(products, key=lambda x: (-float(x['rating']), -x['review_count']))

    # Select the top product and five additional recommendations
    top_product = sorted_products[0]
    additional_recommendations = sorted_products[1:6]

    # Create a structured response before passing to LLM
    recommendation_text = (
        f"Hey, this is the product I would recommend to you:\n\n"
        f" **{top_product['title']}**\n"
        f"   - Price: {top_product['price']}\n"
        f"   - Rating: {top_product['rating']} stars ({top_product['review_count']} reviews)\n"
        f"   - Link: {top_product['link']}\n\n"
        f"Here are five other recommendations:\n"
    )
    
    for i, product in enumerate(additional_recommendations, start=1):
        recommendation_text += (
            f"{i}. **{product['title']}**\n"
            f"   - Price: {product['price']}\n"
            f"   - Rating: {product['rating']} stars ({product['review_count']} reviews)\n"
            f"   - Link: {product['link']}\n\n"
        )

    # Use the LLM to generate a conversational tone if necessary
    prompt = f"Make the following product recommendations sound conversational and friendly:\n\n{recommendation_text}"
    response = llm.generate(prompts=[prompt])

    if response and response.generations and len(response.generations) > 0:
        return response.generations[0][0].text.strip()
    else:
        return recommendation_text  # Fall back to the structured response if LLM fails
