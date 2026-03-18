import asyncio
import aioboto3
from botocore.exceptions import BotoCoreError, ClientError
import dotenv
import os
dotenv.load_dotenv()  

async def retrieve_from_knowledge_base(query: str, knowledge_base_id: str = os.getenv("AWS_KNOWLEDGE_BASE_ID")) -> str:
    """
    Retrieve relevant information from AWS Knowledge Base for RAG.
    
    Args:
        query: The question/query to search for
        knowledge_base_id: The AWS Knowledge Base ID
        
    Returns:
        String containing retrieved text chunks or error message
    """
    try:
        session = aioboto3.Session()
        async with session.client("bedrock-agent-runtime", region_name=os.getenv("AWS_REGION")) as client:
            response = await client.retrieve(
                knowledgeBaseId=knowledge_base_id,
                retrievalQuery={
                    'text': query
                }
            )
            
            # Extract text content from retrieval results
            retrieval_results = response.get('retrievalResults', [])
            
            if not retrieval_results:
                return "No relevant data found in Knowledge Base"
            
            # Combine all text chunks with their scores
            context_chunks = []
            for idx, result in enumerate(retrieval_results, 1):
                content = result.get('content', {})
                text = content.get('text', '')
                score = result.get('score', 0.0)
                
                if text:
                    context_chunks.append(f"[Chunk {idx} - Relevance: {score:.3f}]\n{text}")
            
            if not context_chunks:
                return "No relevant data found in Knowledge Base"
            
            # Join all chunks with separators
            return "\n\n---\n\n".join(context_chunks)
            
    except (BotoCoreError, ClientError) as e:
        print(f"AWS Error: {str(e)}")
        return "Failed to retrieve data from KB"
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return "Failed to retrieve data from KB"


# Test the function
async def main():
    result = await retrieve_from_knowledge_base("What does your company do?")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())