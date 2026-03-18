from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    user_uuid: str
    msg: str

class ChatResponse(BaseModel):
    user_uuid: str
    bot_message: Optional[str] = None
    error: Optional[str] = None