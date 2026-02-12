from openai import OpenAI
import os
import json
from typing import List, Dict, Any, Optional

# Try to import BeautifulSoup for cleaning
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

def clean_content(html_content: str) -> str:
    """
    Cleans HTML content by removing boilerplate tags (nav, header, footer, scripts).
    Returns text content.
    """
    if not BeautifulSoup:
        return html_content # Fallback

    soup = BeautifulSoup(html_content, "lxml")
    
    # Remove distracting elements
    for tag in soup(["script", "style", "nav", "header", "footer", "iframe", "svg", "noscript", "meta"]):
        tag.decompose()
        
    # Get text, or return cleaned HTML if we want structure? 
    # LLMs handle HTML structure well for tables. Let's return cleaned HTML string mainly for structure.
    # But to save tokens, let's try to get a structured text representation or just the body.
    
    body = soup.find("body")
    if body:
        return str(body)[:15000] # Limit to ~15k chars to avoid token limits (approx 3-4k tokens)
    
    return str(soup)[:15000]

def extract_interactive_elements(html_content: str) -> str:
    """
    Extracts interactive elements (a, button, input) with their attributes to help LLM find pagination.
    """
    if not BeautifulSoup or not html_content:
        return ""
        
    soup = BeautifulSoup(html_content, "lxml")
    elements = []
    
    # Find all potentially interactive elements
    for tag in soup.find_all(["a", "button", "input"]):
        # Get key attributes
        attrs = []
        if tag.name == "a" and tag.get("href"):
            attrs.append(f'href="{tag.get("href")}"')
        
        for attr in ["id", "class", "aria-label", "title", "name", "value", "type"]:
             val = tag.get(attr)
             if val:
                 if isinstance(val, list): val = " ".join(val)
                 attrs.append(f'{attr}="{val}"')
        
        text = tag.get_text(strip=True)[:50] # Limit text length
        attr_str = " ".join(attrs)
        
        elements.append(f'<{tag.name} {attr_str}>{text}</{tag.name}>')
        
    return "\n".join(elements[:500]) # Limit to first 500 elements or so to save context

def extract_data_with_llm(content_markdown: str, html_content: str, query: str) -> Dict[str, Any]:
    """
    Extracts company data and next page URL/Selector using LLM.
    Uses Markdown for content and HTML snippets for pagination.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Warning: No OpenAI API Key. Returning empty extraction.")
        return {"companies": [], "next_page_url": None, "pagination_selector": None}

    client = OpenAI(api_key=api_key)
    
    # 1. Prepare Content (Markdown for Companies)
    # 2. Prepare Interactive Elements (HTML for Pagination)
    interactive_html = extract_interactive_elements(html_content)
    
    system_prompt = """You are an expert web scraper extracting SUPPLIER/MANUFACTURER company information from directories, B2B platforms, and listing pages.
    
    YOUR PRIMARY TASK: Extract ALL legitimate businesses from the content that match the user's query.
    
    HOW TO IDENTIFY COMPANIES IN LISTINGS:
    
    1. BUSINESS DIRECTORIES & B2B PLATFORMS:
       - Look for structured lists, tables, or repeated patterns of company information
       - Each entry typically has: company name, and MAY have contact details (website, email, phone, address)
       - Extract EVERY company entry you find, even if contact info is incomplete
       - Common patterns: "Company Name | City | Product" or table rows with company details
    
    2. COMPANY PROFILE PAGES:
       - Extract the main company's information
       - Include all available contact details
    
    3. WHAT TO EXTRACT:
       For each company found:
       - "name": Official business/company name (NOT page titles like "Home" or "About Us")
       - "website": Company website URL (if available)
       - "email": Contact email (if available)
       - "phone": Phone number (if available)
       - "address": Physical address or location (city/country at minimum if available)
       - "description": Brief description of products/services (if available)
       
       NOTE: It's OK if some fields are null/empty. Extract the company if you have at least the name and it's clearly a business.
    
    DO NOT EXTRACT (Critical - these are NOT companies):
    ❌ BUYER REQUESTS: Posts asking for suppliers (e.g., "I want supply of...", "Anyone selling...", "Looking for...")
    ❌ ERROR PAGES: Cloudflare errors, 404, 403, "Access Denied", "Page Not Found"
    ❌ NAVIGATION: "About Us", "Contact Us", "Home", "Login", "Sign Up" (unless these are actual company names in a listing)
    ❌ FORUM QUESTIONS: Q&A posts, discussion threads asking for recommendations
    ❌ PAGE METADATA: Generic website names, categories, or section headings used as company names
    
    VALIDATION CHECKLIST:
    ✓ Is this a real business name? (Not a page title, not a question, not a request)
    ✓ Does it supply/manufacture/distribute products or services?
    ✓ If it's from a listing/directory, extract it even with minimal details
    ✓ Skip if it's clearly a buyer looking for suppliers
    ✓ Skip if it's an error page or navigation element
    
    EXAMPLES OF WHAT TO EXTRACT:
    ✅ "ABC Cosmetics Ltd - Kathmandu - info@abc.com.np - +977-1-4567890"
    ✅ "XYZ Trading Company" (even without contact info, if from a business directory)
    ✅ Table rows listing: Company Name | Location | Product Category
    ✅ Company cards/listings with partial information
    
    EXAMPLES OF WHAT NOT TO EXTRACT:
    ❌ "I need cosmetics suppliers in Nepal" (buyer request)
    ❌ "Cloudflare" (error page)
    ❌ "Home" or "About Us" (navigation)
    ❌ "Looking for manufacturers" (question/request)
    
    Task 2: Identify Pagination
    Look for 'Next', '>', 'Load More', or page numbers in the HTML snippets.
    - 'next_page_url': URL from href
    - 'pagination_selector': CSS selector (e.g. 'a.next-page')
    
    Return JSON:
    {
        "companies": [...],  // Extract ALL legitimate businesses found
        "next_page_url": "...",
        "pagination_selector": "..."
    }
    """
    
    user_prompt = f"""User Query: {query}
    
    --- MARKDOWN CONTENT (For Companies) ---
    {content_markdown[:15000]}
    
    --- INTERACTIVE HTML SNIPPETS (For Pagination) ---
    {interactive_html[:10000]}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-16k", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
        
    except Exception as e:
        print(f"LLM Extraction Error: {e}")
        return {"companies": [], "next_page_url": None, "pagination_selector": None}
