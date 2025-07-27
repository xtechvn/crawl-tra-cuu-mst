from pydantic import BaseModel
from typing import Optional, Dict, Any

class SearchResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: str
