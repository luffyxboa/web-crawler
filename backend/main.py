import asyncio
import json
import csv
import io
import os
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from models import SearchRequest, Company, EnrichRequest
from services.searxng import search_google
from services.crawler import process_url_flow

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store results in memory temporarily for download (In production, use a DB)
last_search_results: List[Company] = []

@app.get("/")
async def health_check():
    """Health check endpoint for Railway"""
    return {"status": "ok", "message": "Web Crawler API is running"}


@app.post("/search")
async def search_endpoint(request: SearchRequest):
    """
    Orchestrates the search and crawl process.
    Streams events: "status", "url_found", "company_extracted", "done".
    """
    query = request.query
    country = request.country
    
    async def event_generator():
        global last_search_results
        last_search_results = []
        
        # 1. Search
        # Build search query with country if provided
        search_query = f"{query} {country}" if country else query
        yield f"data: {json.dumps({'type': 'status', 'message': f'Searching for: {search_query}'})}\n\n"
        
        # Run SearxNG in a thread pool (since requests is blocking)
        search_results = await asyncio.to_thread(search_google, search_query, request.limit)
        
        if not search_results:
            yield f"data: {json.dumps({'type': 'status', 'message': 'No URLs found from search'})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        yield f"data: {json.dumps({'type': 'status', 'message': f'Found {len(search_results)} candidates. filtering with LLM...'})}\n\n"
        
        # Filter with LLM
        from services.llm_filter import filter_search_results
        urls = await asyncio.to_thread(filter_search_results, search_results, search_query)
        
        yield f"data: {json.dumps({'type': 'status', 'message': f'LLM selected {len(urls)} relevant URLs. Starting crawl...'})}\n\n"
        
        # 2. Process each URL
        for url in urls:
            yield f"data: {json.dumps({'type': 'status', 'message': f'Checking URL: {url}'})}\n\n"
            
            # Check Relevance & Extract
            try:
                companies = await asyncio.to_thread(process_url_flow, url, search_query)
                
                if companies:
                     yield f"data: {json.dumps({'type': 'status', 'message': f'Found {len(companies)} companies on {url}'})}\n\n"
                     for company in companies:
                         last_search_results.append(company)
                         yield f"data: {json.dumps({'type': 'company', 'data': company.dict()})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'status', 'message': f'Skipped or no data: {url}'})}\n\n"

            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': f'Error crawling {url}: {str(e)}'})}\n\n"
        
        yield f"data: {json.dumps({'type': 'status', 'message': f'✅ Crawling completed! Found {len(last_search_results)} companies total.'})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/download/{format}")
def download_results(format: str):
    """
    Download the last search results as CSV or JSON.
    """
    if format == "json":
        data = json.dumps([c.dict() for c in last_search_results], indent=2)
        return StreamingResponse(io.StringIO(data), media_type="application/json", headers={"Content-Disposition": "attachment; filename=companies.json"})
    
    elif format == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Name", "Website", "Description", "Email", "Phone", "Address", "Source URL"])
        for c in last_search_results:
            writer.writerow([c.name, c.website, c.description, c.email, c.phone, c.address, c.source_url])
        output.seek(0)
        return StreamingResponse(output, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=companies.csv"})
    
    return {"error": "Invalid format"}


@app.post("/enrich")
async def enrich_endpoint(request: EnrichRequest):
    """
    Enrichment pipeline: deduplicate, search, and enrich contact details.
    Streams progress updates.
    """
    async def event_generator():
        global last_search_results
        
        # Convert dicts to Company objects
        from services.enrichment import deduplicate_by_name, enrich_company_details
        from services.searxng import search_google
        
        companies = [Company(**c) for c in request.companies]
        
        yield f"data: {json.dumps({'type': 'status', 'message': f'Starting enrichment for {len(companies)} companies...'})}\\n\\n"
        
        # Step 1: Deduplicate
        unique_companies = deduplicate_by_name(companies)
        yield f"data: {json.dumps({'type': 'status', 'message': f'Deduplicated to {len(unique_companies)} unique companies'})}\\n\\n"
        
        # Step 2 & 3: Enrich each
        enriched_companies = []
        
        for idx, company in enumerate(unique_companies):
            yield f"data: {json.dumps({'type': 'status', 'message': f'Enriching {idx+1}/{len(unique_companies)}: {company.name}'})}\\n\\n"
            
            # Search for company contact info (including country if provided)
            if request.country:
                search_query = f"{company.name} {request.country} contact information"
            else:
                search_query = f"{company.name} contact information"
            search_results = await asyncio.to_thread(search_google, search_query, 40)
            snippets = [result.get("content", "") for result in search_results if result.get("content")]
            
            if not snippets:
                yield f"data: {json.dumps({'type': 'status', 'message': f'  No search results for {company.name}'})}\\n\\n"
                enriched_companies.append(company)
                continue
            
            # LLM enrichment
            enriched_data = await asyncio.to_thread(enrich_company_details, company.name, snippets)
            
            # Merge enriched data
            company.email = enriched_data.get("email") or company.email
            company.phone = enriched_data.get("phone") or company.phone
            company.address = enriched_data.get("address") or company.address
            company.website = enriched_data.get("website") or company.website
            company.description = enriched_data.get("description") or company.description
            
            enriched_companies.append(company)
            yield f"data: {json.dumps({'type': 'company', 'data': company.dict()})}\\n\\n"
        
        # Update global results
        last_search_results = enriched_companies
        
        yield f"data: {json.dumps({'type': 'done', 'message': f'Enrichment complete! {len(enriched_companies)} companies enriched'})}\\n\\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
