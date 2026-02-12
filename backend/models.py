from pydantic import BaseModel
from typing import List, Optional

class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    country: Optional[str] = None  # Optional country filter for more targeted searches

class Company(BaseModel):
    name: str
    website: Optional[str] = None
    description: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    source_url: str

class CrawlStatus(BaseModel):
    url: str
    status: str  # "crawling", "skipped", "completed", "failed"
    companies_found: int = 0
    message: Optional[str] = None

class SearchResponse(BaseModel):
    results: List[Company]
    total_companies: int

class EnrichRequest(BaseModel):
    companies: List[dict]  # List of company dicts from frontend
    country: Optional[str] = None  # Optional country for more targeted enrichment
