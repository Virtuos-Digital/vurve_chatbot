"""
Vector search utility for product recommendations using pgvector and HNSW.
"""
import os
import httpx
import asyncpg
import numpy as np
from typing import List, Dict, Any
from dotenv import load_dotenv
from src.logger import LOG_TYPES, logger

# Load environment variables
load_dotenv()


async def get_embedding(text: str) -> List[float]:
    """
    Convert text to vector embedding using the hosted model.
    
    Args:
        text: Input text to embed
        
    Returns:
        List of floats representing the embedding vector
    """
    try:
        embed_url = os.getenv("EMBED_API_URL", "https://gift-intelligence.vurve.ai/api/v1/embed")
        
        payload = {
            "texts": [text],
            "normalize": True
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(embed_url, json=payload)
            response.raise_for_status()
            data = response.json()
            
        # Extract the embedding from the response
        # API returns: {"embeddings": [{"text": "...", "embedding": [0.1, 0.2, ...]}]}
        embeddings = data.get("embeddings", [])
        if not embeddings:
            logger(f"No embeddings returned from API", LOG_TYPES.ERROR)
            return []
        
        # Extract the embedding array from the first embedding object
        embedding_obj = embeddings[0]
        if isinstance(embedding_obj, dict):
            embedding = embedding_obj.get("embedding", [])
        else:
            # Fallback if API format changes
            embedding = embedding_obj
            
        if not embedding:
            logger(f"No embedding array found in response", LOG_TYPES.ERROR)
            return []
            
        logger(f"Generated embedding for text (dimension: {len(embedding)})", LOG_TYPES.SUCCESS)
        return embedding
        
    except httpx.RequestError as e:
        logger(f"Failed to get embedding from API: {e}", LOG_TYPES.ERROR)
        return []
    except Exception as e:
        logger(f"Unexpected error getting embedding: {e}", LOG_TYPES.ERROR)
        return []


async def vector_search_products(query_text: str, top_k: int = 5) -> List[str]:
    """
    Perform vector similarity search for products using HNSW index.
    
    Args:
        query_text: The search query text
        top_k: Number of top results to return (default: 5)
        
    Returns:
        List of shopify_product_id strings for the top matching products
    """
    try:
        # Get embedding for query text
        query_embedding = await get_embedding(query_text)
        if not query_embedding:
            logger("Failed to generate query embedding", LOG_TYPES.ERROR)
            return []
        
        # Connect to vector database
        db_host = os.getenv("VECTOR_DB_HOST")
        db_port = int(os.getenv("VECTOR_DB_PORT"))
        db_name = os.getenv("VECTOR_DB_NAME")
        db_user = os.getenv("VECTOR_DB_USER")
        db_password = os.getenv("VECTOR_DB_PASSWORD")
        
        # Build connection kwargs (handle empty password)
        conn_kwargs = {
            "host": db_host,
            "port": db_port,
            "database": db_name,
            "user": db_user
        }
        if db_password:
            conn_kwargs["password"] = db_password
        
        conn = await asyncpg.connect(**conn_kwargs)
        
        try:
            # Perform vector similarity search using HNSW index
            # Convert embedding to pgvector format
            embedding_str = str(query_embedding)
            
            # Query using cosine distance (<=> operator for cosine distance in pgvector)
            query = """
                SELECT shopify_product_id
                FROM embedding_minilm
                ORDER BY embedding <=> $1::vector
                LIMIT $2
            """
            
            rows = await conn.fetch(query, embedding_str, top_k)
            
            # Extract shopify_product_ids
            product_ids = [row['shopify_product_id'] for row in rows]
            
            logger(f"Found {len(product_ids)} matching products for query: '{query_text}'", LOG_TYPES.SUCCESS)
            return product_ids
            
        finally:
            await conn.close()
            
    except asyncpg.PostgresError as e:
        logger(f"Database error during vector search: {e}", LOG_TYPES.ERROR)
        return []
    except Exception as e:
        logger(f"Unexpected error during vector search: {e}", LOG_TYPES.ERROR)
        return []


async def test_vector_search():
    """Test the vector search functionality."""
    logger("Testing vector search...", LOG_TYPES.INFORMATION)
    
    # Test embedding generation
    test_text = "red dress for party"
    embedding = await get_embedding(test_text)
    if embedding:
        logger(f"Embedding test passed - dimension: {len(embedding)}", LOG_TYPES.SUCCESS)
    else:
        logger("Embedding test failed", LOG_TYPES.ERROR)
        return
    
    # Test vector search
    results = await vector_search_products(test_text, top_k=5)
    if results:
        logger(f"Vector search test passed - found {len(results)} products", LOG_TYPES.SUCCESS)
        logger(f"Product IDs: {results}", LOG_TYPES.INFORMATION)
    else:
        logger("Vector search test returned no results", LOG_TYPES.WARNING)


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_vector_search())
