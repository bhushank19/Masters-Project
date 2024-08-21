import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import diskcache as dc  # Import DiskCache

cache = dc.Cache('/tmp/web_scraping_cache')  # Create a cache instance

# Web scraping functions
async def fetch_page_content(url):
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url)
            content = await page.content()
            await browser.close()
            return content
    except Exception as e:
        print(f"Error fetching page content: {e}")
        return None

async def cached_fetch_page_content(url):
    if url in cache:
        return cache[url]
    else:
        content = await fetch_page_content(url)  # Await the async function
        cache[url] = content
        return content

import logging

import logging
from bs4 import BeautifulSoup

def parse_amazon_product_list(html_content):
    if html_content is None:
        return []

    soup = BeautifulSoup(html_content, 'html.parser')
    products = []
    
    for product in soup.select('.s-main-slot .s-result-item'):
        title = product.select_one('h2 a span')
        price_whole = product.select_one('.a-price-whole')
        price_fraction = product.select_one('.a-price-fraction')
        rating = product.select_one('.a-icon-alt')
        review_count = product.select_one('.a-size-base')  # Selector for review count
        image = product.select_one('.s-image')
        product_link = product.select_one('h2 a')['href'] if product.select_one('h2 a') else None
        
        if title and price_whole and rating and image and product_link and review_count:
            price = f"{price_whole.text}.{price_fraction.text if price_fraction else '00'}"
            
            # Validate and extract review count
            try:
                review_count_value = int(review_count.text.replace(",", ""))
            except ValueError:
                review_count_value = 0  # Set to 0 if invalid review count
            
            product_details = {
                "title": title.text.strip(),
                "description": 'No description',  # We can add description parsing later
                "price": price,
                "rating": rating.text.strip().split(" ")[0],  # Extract the numeric rating
                "review_count": review_count_value,  # Use the validated review count
                "image": image['src'],
                "link": f"https://www.amazon.com{product_link}"
            }
            products.append(product_details)
    return products



async def fetch_multiple_pages(base_url, num_pages=2):
    tasks = []
    for page_number in range(1, num_pages + 1):
        url = f"{base_url}&page={page_number}"
        tasks.append(cached_fetch_page_content(url))

    # Fetch all pages concurrently
    pages_content = await asyncio.gather(*tasks)

    all_products = []
    for content in pages_content:
        products = parse_amazon_product_list(content)
        all_products.extend(products)
    
    return all_products

async def fetch_product_data(user_query: str, num_pages=2):
    base_url = f"https://www.amazon.com/s?k={user_query.replace(' ', '+')}"
    products = await fetch_multiple_pages(base_url, num_pages)
    
    # Log the number of products fetched and inspect the data
    print(f"Number of products fetched: {len(products)}")
    for product in products[:5]:  # Print first 5 products for inspection
        print(product)
    
    return products

