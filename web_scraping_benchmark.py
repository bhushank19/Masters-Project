import time
import asyncio
from web_scraping import fetch_product_data

# Function to benchmark web scraping
async def benchmark_web_scraping(user_queries, num_pages=2):
    # Initialize benchmarking results
    benchmark_results = []

    for query in user_queries:
        print(f"Benchmarking for query: {query}")

        # Measure start time
        start_time = time.time()

        # Fetch product data
        products = await fetch_product_data(query, num_pages)

        # Measure end time
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Calculate metrics
        num_products = len(products)
        errors = sum(1 for product in products if not product.get('title') or not product.get('price'))  
        error_rate = errors / num_products if num_products > 0 else 0

        # Log the results for this query
        benchmark_results.append({
            'query': query,
            'num_products': num_products,
            'errors': errors,
            'error_rate': error_rate,
            'scraping_time': elapsed_time
        })

    return benchmark_results

# Example usage
if __name__ == "__main__":
    user_queries = [
        "Samsung Galaxy S23",
        "Apple MacBook Pro",
        "Sony WH-1000XM5",
        "Dell XPS 13 laptop",
        "Logitech Mouse"
    ]
    
    # Run benchmarking
    benchmark_results = asyncio.run(benchmark_web_scraping(user_queries, num_pages=2))

    # Print benchmarking results
    print("\nWeb Scraping Benchmarking Results:")
    for result in benchmark_results:
        print(f"Query: {result['query']}")
        print(f"Number of Products Scraped: {result['num_products']}")
        print(f"Errors: {result['errors']}")
        print(f"Error Rate: {result['error_rate']:.2%}")
        print(f"Scraping Time: {result['scraping_time']:.2f} seconds")
        print("-" * 40)

