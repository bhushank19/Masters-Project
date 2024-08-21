import time
from rag_integration import refine_query_with_llm, filter_irrelevant_products, generate_recommendations_with_llm
from web_scraping import fetch_product_data
import asyncio

# Define a set of generic queries for benchmarking
queries = [
    "Find me the latest Samsung Galaxy phone",
    "Recommend me a powerful Apple laptop",
    "Show me the best Sony noise-cancelling headphones",
    "What's the latest Dell laptop for professionals?",
    "Suggest a good wireless mouse from Logitech"
]

# Performance metrics storage
llm_performance_results = []

# Function to measure query refinement
async def benchmark_query_refinement(query):
    print(f"Original Query: {query}")
    start_time = time.time()
    refined_query = await refine_query_with_llm(query)
    refinement_time = time.time() - start_time
    print(f"Refined Query: {refined_query}\n")
    
    # Log the results
    llm_performance_results.append({
        "Query": query,
        "Refined Query": refined_query,
        "Refinement Time (s)": refinement_time
    })
    return refined_query

# Function to measure product filtering
async def benchmark_product_filtering(refined_query):
    # Fetch products
    products = await fetch_product_data(refined_query)
    if not products:
        print(f"No products found for query: {refined_query}")
        return
    
    # Perform product filtering
    start_time = time.time()
    filtered_products = await filter_irrelevant_products(products, refined_query)
    filtering_time = time.time() - start_time
    
    # Log the results
    llm_performance_results.append({
        "Refined Query": refined_query,
        "Number of Products Filtered": len(filtered_products),
        "Filtering Time (s)": filtering_time
    })
    return filtered_products

# Function to measure recommendation generation
async def benchmark_recommendation_generation(refined_query, filtered_products):
    # Perform recommendation generation
    start_time = time.time()
    recommendations = await generate_recommendations_with_llm(filtered_products)
    recommendation_time = time.time() - start_time
    
    # Log the results
    llm_performance_results.append({
        "Refined Query": refined_query,
        "Number of Recommendations Generated": len(recommendations.split('\n\n')) - 1,  # Split on double newline
        "Recommendation Time (s)": recommendation_time
    })

# Main benchmark function
async def benchmark_llm():
    for query in queries:
        # Step 1: Query Refinement
        refined_query = await benchmark_query_refinement(query)
        
        # Step 2: Product Filtering
        filtered_products = await benchmark_product_filtering(refined_query)
        if not filtered_products:
            continue  # Skip recommendation generation if no products are found
        
        # Step 3: Recommendation Generation
        await benchmark_recommendation_generation(refined_query, filtered_products)
    
    # Display final results
    print("\nLLM Benchmarking Results:")
    for result in llm_performance_results:
        print(result)

# Run the benchmark
if __name__ == "__main__":
    asyncio.run(benchmark_llm())
