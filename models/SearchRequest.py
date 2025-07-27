from pydantic import BaseModel
from typing import Optional

class SearchRequest(BaseModel):
    tax_code: str
    company_name: Optional[str] = None