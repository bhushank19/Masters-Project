import streamlit as st
from web_scraping import fetch_product_data
from rag_integration import refine_query_with_llm, filter_irrelevant_products, rag_workflow
import asyncio

# Streamlit frontend
st.title("AI-based Product Recommendations")

# Maintain state for fetched products
if 'products' not in st.session_state:
    st.session_state['products'] = []

query = st.text_input("Describe the product you are looking for:")

if st.button("Get Recommendations"):
    if query:
        with st.spinner('Refining your query...'):
            # Step 1: Refine the query using LLM
            refined_query = asyncio.run(refine_query_with_llm(query))
            st.write(f"Refined Query: {refined_query}")
        
        with st.spinner('Fetching product data...'):
            # Step 2: Fetch product data from Amazon
            products = asyncio.run(fetch_product_data(refined_query))
            
            with st.spinner('Filtering irrelevant products...'):
                # Step 3: Filter irrelevant products using LLM
                filtered_products = asyncio.run(filter_irrelevant_products(products, refined_query))
                st.session_state['products'] = filtered_products

products = st.session_state['products']

if products:
    with st.spinner('Generating AI-driven recommendations...'):
        # Step 4: Set up RAG and generate recommendations
        recommendations = asyncio.run(rag_workflow(query, products))
        st.write(recommendations)
else:
    st.write("No relevant products found.")
