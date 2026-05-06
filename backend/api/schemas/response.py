from pydantic import BaseModel
from typing import Optional

class QueryResponse(BaseModel):
    answer: str
    query: Optional[str] = None  # populated by voice endpoint with the transcribed text