# Imports
import os
import httpx
import asyncio 
import copy
import uuid
import json
from typing import TypedDict, Any
from dotenv import load_dotenv  
from langgraph.graph import StateGraph, END, START
from src.logger import LOG_TYPES, logger
from src.redis_utils import test_redis_connection, get_redis_client, store_user_message_to_redis, get_user_chat_context
from src.types.api import ChatResponse
load_dotenv()

if "TEST_VARIABLE" in os.environ:
    logger(".env is working", LOG_TYPES.SUCCESS)
else:    
    logger(".env is not working", LOG_TYPES.ERROR)

test_redis_connection()
redis_client = get_redis_client()

class AgentState(TypedDict):
    """State of the agent."""
    user_uuid: str
    last_user_message: str
    data: dict

async def classify_message_intent(state: AgentState) -> AgentState:
    """Classify the intent of the message."""

    new_state: AgentState = copy.deepcopy(state)
    
    try:
        store_user_message_to_redis(
            redis_client,
            state["user_uuid"], 
            state["last_user_message"], 
            "user_message"
        )

        url = os.getenv("INTENT_CLASSIFIER_API_URL")
        if not url:
            logger("Error: INTENT_CLASSIFIER_API_URL not set", LOG_TYPES.ERROR)
            new_state['data']['classification_error'] = "INTENT_CLASSIFIER_API_URL not set"
            return new_state
        
        # Prepare the payload
        logger(state["last_user_message"], LOG_TYPES.INFORMATION)
        payload = {"sentence": state["last_user_message"]}
        
        # Make the POST request
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        description = data.get("description")
        if not description:
            logger("Error: Invalid response format (missing 'description')", LOG_TYPES.ERROR)
            new_state['data']['classification_error'] = "Invalid response format"
            return new_state

        logger(f"Message: '{state['last_user_message']}' classified as: {description}", LOG_TYPES.SUCCESS)

        new_state['data']['classified_intent'] = description
        return new_state
        
    except httpx.RequestError as e:
        logger(f"API request failed: {e}", LOG_TYPES.ERROR)
        new_state['data']['classification_error'] = "API request failed"
        return new_state
    except httpx.HTTPStatusError as e:
        logger(f"API returned error status: {e.response.status_code}", LOG_TYPES.ERROR)
        new_state['data']['classification_error'] = "API request failed"
        return new_state
    except ValueError:
        logger("Response was not valid JSON", LOG_TYPES.ERROR)
        new_state['data']['classification_error'] = "Invalid JSON response"
        return new_state
    except Exception as e:
        logger(f"Unexpected error during intent classification: {e}", LOG_TYPES.ERROR)
        new_state['data']['classification_error'] = "Unexpected error during intent classification"
        return new_state

def route_after_classification(state: AgentState) -> str:
    """Route based on classification result."""
    if 'classification_error' in state['data']:
        return "error"
    
    classified_intent = state['data'].get('classified_intent')
    return classified_intent
    
async def recommend_product(state: AgentState) -> AgentState:
    """Recommend a product based on the message."""

    new_state: AgentState = copy.deepcopy(state)
    
    logger(f"Recommending product for message: '{state['last_user_message']}'", LOG_TYPES.INFORMATION)
    
    # Generate product recommendation based on message
    bot_message = f"I recommend checking out our premium products for '{state['last_user_message']}'. Here are some great options!"
    
    # Store the bot message in state for response
    new_state['data']['bot_message'] = bot_message
    
    try:
        # Store the bot response to Redis as well
        store_user_message_to_redis(
            redis_client,
            state["user_uuid"], 
            bot_message, 
            "bot_response"
        )

        logger(f"Product recommendation generated for user: '{state['user_uuid']}'", LOG_TYPES.SUCCESS)
    except Exception as e:
        logger(f"Failed to store bot response to Redis: {e}", LOG_TYPES.ERROR)
    
    return new_state

def send_response_to_user(state: AgentState) -> AgentState:
    """Send HTTP response back to the user via FastAPI Response object."""
    
    user_uuid = state['user_uuid']
    
    try:
        # Determine response based on state data using ChatResponse model
        if 'classification_error' in state['data']:
            # Handle classification errors - create ChatResponse with error
            chat_response = ChatResponse(
                user_uuid=user_uuid,
                error=state['data']['classification_error']
            )
            status_code = 400
            
            logger(f"Sending error response to user {user_uuid}: {state['data']['classification_error']}", LOG_TYPES.WARNING)
            
        elif 'bot_message' in state['data']:
            # Handle successful bot responses - create ChatResponse with bot_message
            chat_response = ChatResponse(
                user_uuid=user_uuid,
                bot_message=state['data']['bot_message']
            )
            status_code = 200
            
            logger(f"Sending bot response to user {user_uuid}", LOG_TYPES.SUCCESS)
            
        else:
            # Default fallback for unhandled cases - create ChatResponse with error
            chat_response = ChatResponse(
                user_uuid=user_uuid,
                error="No response generated"
            )
            status_code = 500
            
            logger(f"Sending fallback error response to user {user_uuid}", LOG_TYPES.WARNING)
        
        # Store response data in state for FastAPI to return
        new_state = copy.deepcopy(state)
        new_state['data']['response_data'] = chat_response.model_dump(exclude_none=True)
        new_state['data']['response_status_code'] = status_code
        
        logger(f"Response prepared: {status_code} - {chat_response.model_dump(exclude_none=True)}", LOG_TYPES.SUCCESS)
        return new_state
        
    except Exception as e:
        logger(f"Failed to prepare HTTP response: {e}", LOG_TYPES.ERROR)
        
        # Set emergency fallback response using ChatResponse model
        error_response = ChatResponse(
            user_uuid=user_uuid,
            error="Internal server error while formatting response"
        )
        
        new_state = copy.deepcopy(state)
        new_state['data']['response_data'] = error_response.model_dump(exclude_none=True)
        new_state['data']['response_status_code'] = 500
            
        logger("Internal Servor error in langgraph node 'send_response_to_user'", LOG_TYPES.ERROR)
        return new_state

graph = StateGraph(AgentState)
graph.add_node("intent_classifier", classify_message_intent)
graph.add_node("product_recommender", recommend_product)
graph.add_node("send_response_to_user", send_response_to_user)

graph.add_edge(START, "intent_classifier")
graph.add_conditional_edges(
    "intent_classifier",
    route_after_classification,
    {
        "Product recommendation": "product_recommender",
        "error": "send_response_to_user"
    }
)
graph.add_edge("product_recommender", "send_response_to_user")
graph.add_edge("send_response_to_user", END)

bot = graph.compile()

if __name__ == "__main__":
    # Test the chatbot with proper user session
    # test_user_uuid = str(uuid.uuid4())
    test_user_uuid = "e2ae5050-44b7-4e44-badc-7c48c799ac96" # Using a fixed UUID for testing
    logger(f"Testing with user UUID: {test_user_uuid}", LOG_TYPES.INFORMATION)

    input = {
        "user_uuid": test_user_uuid,
        "last_user_message": "Which Samsung should I buy?",
        "data": {}
    }

    result = asyncio.run(bot.ainvoke(input))
    logger(f"Bot result: {result}", LOG_TYPES.INFORMATION)

    # Get chat context after interaction
    context = get_user_chat_context(redis_client, test_user_uuid)
    logger(f"Chat context after interaction:", LOG_TYPES.INFORMATION)
    for message in context:
        logger(f"  {message}", LOG_TYPES.INFORMATION)