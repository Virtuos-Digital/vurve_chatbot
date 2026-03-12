# Imports
import os
import httpx
import asyncio 
import copy
import uuid
import json
from typing import TypedDict, Any
from dotenv import load_dotenv  
from openai import OpenAI
from langgraph.graph import StateGraph, END, START
from src.logger import LOG_TYPES, logger
from src.redis_utils import test_redis_connection, get_redis_client, store_user_message_to_redis, get_user_chat_context
from src.types.api import ChatResponse
load_dotenv()

if "TEST_VARIABLE" in os.environ:
    logger(".env is working", LOG_TYPES.SUCCESS)
else:    
    logger(".env is not working", LOG_TYPES.ERROR)

# Redis client will be initialized lazily in async context
redis_client = None

async def initialize_redis():
    """Initialize Redis client asynchronously. Returns the Redis client."""
    global redis_client
    if redis_client is None:
        await test_redis_connection()
        redis_client = await get_redis_client()
    return redis_client

class AgentState(TypedDict):
    """State of the agent."""
    user_uuid: str
    last_user_message: str
    data: dict

async def classify_message_intent(state: AgentState) -> AgentState:
    """Classify the intent of the message."""

    new_state: AgentState = copy.deepcopy(state)
    
    try:
        # Ensure Redis client is initialized
        redis_client = await initialize_redis()
        
        stored_to_redis = await store_user_message_to_redis(
            redis_client,
            state["user_uuid"], 
            state["last_user_message"], 
            "user_message"
        )

        if(not stored_to_redis):
            logger("Failed to store user message to Redis", LOG_TYPES.WARNING)
            raise Exception("Failed to store user message to Redis")

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
        # Ensure Redis client is initialized
        redis_client = await initialize_redis()
        
        # Store the bot response to Redis as well
        await store_user_message_to_redis(
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
    
async def answer_general_query(state: AgentState) -> AgentState:
    """Answer Queries of General FAQ/ Company Policy nature."""
    new_state: AgentState = copy.deepcopy(state)

    try:
        # Ensure Redis client is initialized
        redis_client = await initialize_redis()
        
        # Step 1: Retrieve conversation history from Redis
        context = await get_user_chat_context(redis_client, state["user_uuid"])
        
        if context is None:
            raise Exception("Failed to retrieve context from redis")
        
        # Parse Redis messages into OpenAI message format
        messages = []
        for msg in context:
            if msg.startswith("user_message: "):
                content = msg.replace("user_message: ", "", 1)
                messages.append({"role": "user", "content": content})
            elif msg.startswith("bot_response: "):
                content = msg.replace("bot_response: ", "", 1)
                messages.append({"role": "assistant", "content": content})
        
        # Add system message at the beginning
        messages.insert(0, {
            "role": "system", 
            "content": "You are a helpful customer service assistant. Answer questions about products, company policies, and general inquiries based on the conversation history."
        })
        
        logger(f"Prepared {len(messages)} messages for LLM", LOG_TYPES.INFORMATION)
        
        # Initialize OpenAI client
        openai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
        
        # Step 2: Loop until we get a non-tool response
        max_iterations = 10  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            logger(f"LLM call iteration {iteration}", LOG_TYPES.INFORMATION)
            
            # Send to LLM
            try:
                completion = openai_client.chat.completions.create(
                    model="mistral.ministral-3-8b-instruct",
                    messages=messages,
                    # tools=tools  # Add tools parameter here when tools are defined
                )
            except Exception as e:
                raise Exception(f"AWS bedrock error: {str(e)}")
            
            assistant_message = completion.choices[0].message
            
            # Step 2.1: Check for tool calls
            if assistant_message.tool_calls:
                logger(f"Assistant requested {len(assistant_message.tool_calls)} tool calls", LOG_TYPES.INFORMATION)
                
                # Append assistant message with tool calls
                messages.append(assistant_message)
                
                # Execute each tool call
                for tool_call in assistant_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    logger(f"Executing tool: {function_name} with args: {function_args}", LOG_TYPES.INFORMATION)
                    
                    # TODO: Implement actual tool calling logic here
                    # For now, return a placeholder
                    result = f"Tool {function_name} executed successfully"
                    
                    # Append tool result
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result)
                    })
                
                # Loop back to step 2
                continue
            else:
                # No tool calls, we have the final response
                logger(f"Received final response from LLM", LOG_TYPES.SUCCESS)
                new_state['data']['bot_message'] = assistant_message.content
                
                try:
                    # Store bot response to Redis
                    await store_user_message_to_redis(
                        redis_client,
                        state["user_uuid"],
                        assistant_message.content,
                        "bot_response"
                    )
                except Exception as e:
                    logger(f"Failed to store bot response to Redis: {e}", LOG_TYPES.ERROR)
                    raise Exception("Failed to store bot response to Redis")
                
                break
        
        if iteration >= max_iterations:
            raise Exception("Maximum tool call iterations reached")
            
    except Exception as e:
        logger(f"Error in answer_general_query: {e}", LOG_TYPES.ERROR)
        error_msg = str(e)
        
        if "Failed to retrieve context from redis" in error_msg:
            new_state['data']['error'] = "Failed to retrieve context from redis"
        elif "AWS bedrock error" in error_msg:
            new_state['data']['error'] = error_msg
        elif "Failed to store bot response to Redis" in error_msg:
            new_state['data']['error'] = "Failed to store bot response to Redis"
        else:
            new_state['data']['error'] = f"Error processing query: {error_msg}"
    
    return new_state

graph = StateGraph(AgentState)
graph.add_node("intent_classifier", classify_message_intent)
graph.add_node("product_recommender", recommend_product)
graph.add_node("send_response_to_user", send_response_to_user)
graph.add_node("answer_general_query", answer_general_query)

graph.add_edge(START, "intent_classifier")
graph.add_conditional_edges(
    "intent_classifier",
    route_after_classification,
    {
        "Product recommendation": "product_recommender",
        "General conversation/company policy": "answer_general_query",
        "error": "send_response_to_user"
    }
)
graph.add_edge("product_recommender", "send_response_to_user")
graph.add_edge("answer_general_query", "send_response_to_user")
graph.add_edge("send_response_to_user", END)

bot = graph.compile()

if __name__ == "__main__":
    async def main():
        # Test the chatbot with proper user session
        # test_user_uuid = str(uuid.uuid4())
        test_user_uuid = "e2ae5050-44b7-4e44-badc-7c48c799ac96" # Using a fixed UUID for testing
        logger(f"Testing with user UUID: {test_user_uuid}", LOG_TYPES.INFORMATION)

        input = {
            "user_uuid": test_user_uuid,
            "last_user_message": "Which Samsung should I buy?",
            "data": {}
        }

        result = await bot.ainvoke(input)
        logger(f"Bot result: {result}", LOG_TYPES.INFORMATION)

        # Get chat context after interaction
        redis_client = await initialize_redis()
        context = await get_user_chat_context(redis_client, test_user_uuid)
        logger(f"Chat context after interaction:", LOG_TYPES.INFORMATION)
        for message in context:
            logger(f"  {message}", LOG_TYPES.INFORMATION)
    
    asyncio.run(main())