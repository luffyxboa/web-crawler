import requests
import json
import time
from typing import List, Dict, Any, Optional
from models import Company

CRAWL4AI_URL = "https://crawle.up.railway.app/crawl"

def crawl_page_raw(url: str, js_code: List[str] = None) -> Dict[str, Any]:
    """
    Helper to fetch raw content from Crawl4AI without LLM extraction.
    Supports optional JS execution (e.g. for pagination).
    """
    if js_code is None:
        js_code = []

    # Default scroll down script + any custom scripts
    combined_js = [
        "const scrollDown = async () => { window.scrollBy(0, window.innerHeight); }; scrollDown();"
    ] + js_code

    payload = {
        "urls": [url],
        "priority": 20,
        "browser_type": "chromium",  # Explicitly use Puppeteer/Chromium
        "headless": True,
        "viewport_width": 1920,
        "viewport_height": 1080,
        # No extraction strategy, just fetch content
        "js_code": combined_js, 
        "wait_for": "networkidle",
        "delay_before_return": 3000, # Wait longer for dynamic content
        "page_timeout": 60000,  # 60 second timeout
        "magic": False, # Use False to get raw markdown/html reliably
        "word_count_threshold": 1,
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        response = requests.post(CRAWL4AI_URL, json=payload, timeout=60)
        # response.raise_for_status() # Don't raise immediately, check status code manually
        
        if response.status_code == 200:
            result = response.json()
            if "results" in result and len(result["results"]) > 0:
                # Return the full result object (with markdown and html)
                return result["results"][0]
        else:
            print(f"Crawl failed for {url}: {response.status_code} - {response.text}")
            
        return {}
    except Exception as e:
        print(f"Crawl exception for {url}: {e}")
        return {}

from services.llm_extractor import extract_data_with_llm

def process_url_flow(start_url: str, query: str) -> List[Company]:
    """
    Orchestrates the crawl flow for a single URL using LLM Extraction & Pagination:
    1. Fetch Raw Content (Page 1)
    2. Extract Data & Next Page using LLM
    3. Loop until no next page or limit reached (3 pages)
    """
    companies: List[Company] = []
    current_url = start_url
    pages_crawled = 0
    visited_urls = set()
    
    # Store JS instructions for the NEXT page load
    next_page_js_code = []

    print(f"Processing URL: {start_url}")

    while current_url and pages_crawled < 3:
        # If we are visiting via a normal URL change, check visited.
        # If we are "clicking" (JS) on the same URL, we might technically be on the same URL string 
        # but displaying different content. So visited checks might need to be relaxed if using JS.
        # Strategy: Trust the loop limit.
        
        if not next_page_js_code and current_url in visited_urls:
            print(f"Already visited {current_url} (and no JS action), stopping loop.")
            break
        visited_urls.add(current_url)

        if pages_crawled > 0:
             print(f"Crawling page {pages_crawled + 1}: {current_url}")
        
        # Execute crawl with any pending JS (e.g. click next)
        page_data = crawl_page_raw(current_url, js_code=next_page_js_code)
        
        # Reset JS code after use
        next_page_js_code = []
        
        if not page_data:
            print(f"Failed to fetch content for {current_url}")
            break
            
        # Prefer Markdown for LLM extraction to save tokens and reduce noise
        # If markdown is empty/fail, fallback to HTML
        content_to_analyze = ""
        markdown_data = page_data.get("markdown", "")
        if isinstance(markdown_data, dict):
            content_to_analyze = markdown_data.get("raw_markdown", "")
        elif isinstance(markdown_data, str):
            content_to_analyze = markdown_data
            
        if not content_to_analyze or len(content_to_analyze) < 100:
            # Fallback to HTML
            content_to_analyze = page_data.get("html", "")
            
        print(f"Analyzing content length: {len(content_to_analyze)}")
        
        # LLM Extraction
        # Pass Markdown for content, HTML for pagination
        html_content = page_data.get("html", "")
        extraction_result = extract_data_with_llm(content_to_analyze, html_content, query)
        
        new_companies_data = extraction_result.get("companies", [])
        
        # Check for Pagination Methods
        next_page_url = extraction_result.get("next_page_url")
        pagination_selector = extraction_result.get("pagination_selector")
        
        print(f"EXTRACTOR: Found {len(new_companies_data)} companies.")
        print(f"   Next URL: {next_page_url}")
        print(f"   Pagination Selector: {pagination_selector}")
        
        for c in new_companies_data:
            # Basic validation/cleanup
            if c.get("name"):
                 companies.append(Company(
                    name=c.get("name", "Unknown"),
                    website=c.get("website"),
                    description=c.get("description"),
                    email=c.get("email"),
                    phone=c.get("phone"),
                    address=c.get("address"),
                    source_url=current_url
                ))
        
        # Pagination Logic Priority
        # 1. URL change is most reliable
        if next_page_url and next_page_url != current_url and next_page_url.startswith("http"):
             current_url = next_page_url
             next_page_js_code = [] # Reset JS
             
        # 2. JS Click if no URL change (or explicit selector found)
        elif pagination_selector:
            # We are staying on 'current_url' (or at least starting from it) but executing a click
            # IMPORTANT: We stick to current_url, but next iteration we pass JS.
            # Only do this if we haven't hit limit.
            print(f"Preparing to click selector: {pagination_selector}")
            next_page_js_code = [
                f"const el = document.querySelector('{pagination_selector}'); if(el) {{ el.click(); }} else {{ console.log('Selector not found'); }}",
                "new Promise(r => setTimeout(r, 3000));" # Wait for update
            ]
            # Verify we aren't looping infinitely on the same page with same selector?
            # The 'pages_crawled' limit handles it.
        else:
            current_url = None # Stop loop
            
        pages_crawled += 1
            
    return companies
