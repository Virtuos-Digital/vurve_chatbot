"""
FastAPI Chat Endpoint
"""
import uvicorn
from fastapi import FastAPI, Response
from src.types.api import ChatRequest, ChatResponse
from src.orchestrator import bot
from src.logger import LOG_TYPES, logger

app = FastAPI(
    title="Chatbot API", 
    description="AI Chatbot with LangGraph Control",
    version="1.0.0"
)

@app.post("/api/v1/talk")
async def talk(request: ChatRequest, response: Response):
    """
    Chat endpoint
    """
    
    logger(f"Received chat request from user: {request.user_uuid}", LOG_TYPES.INFORMATION)
    
    try:
        # Hand off to LangGraph workflow
        result = await bot.ainvoke({
            "user_uuid": request.user_uuid,
            "last_user_message": request.msg,
            "data": {}
        })

        # Extract status and response from LangGraph result
        status_code = result.get('data', {}).get('response_status_code', 500)
        response_data = result.get('data', {}).get('response_data')
        
        # Set response headers and status code
        response.status_code = status_code
        response.headers["content-type"] = "application/json"
        
        if response_data:
            logger(f"Returning response: {status_code} - {response_data}", LOG_TYPES.SUCCESS)
            return response_data
        else:
            # Fallback if no response data was generated
            fallback_response = ChatResponse(
                user_uuid=request.user_uuid,
                error="Workflow completed but no response data generated"
            ).model_dump(exclude_none=True)
            
            response.status_code = 500
            logger("Fallback response returned", LOG_TYPES.WARNING)
            return fallback_response
        
    except Exception as e:
        # Last resort error handling
        logger(f"Critical error in chat workflow: {e}", LOG_TYPES.ERROR)
        
        # Use ChatResponse model for consistent error formatting
        error_response = ChatResponse(
            user_uuid=request.user_uuid,
            error=str(e)
        ).model_dump(exclude_none=True)
        
        response.status_code = 500
        response.headers["content-type"] = "application/json"
        
        logger("Server error response returned from /talk", LOG_TYPES.ERROR)
        return error_response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
