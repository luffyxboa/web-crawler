import os
from typing import List, Dict, Any
from models import Company
from services.searxng import search_google
from openai import OpenAI
import json

def deduplicate_by_name(companies: List[Company]) -> List[Company]:
    """
    Deduplicate companies by name (case-insensitive).
    Keeps the first occurrence of each unique name.
    """
    seen_names = set()
    unique_companies = []
    
    for company in companies:
        name_lower = company.name.lower().strip()
        if name_lower and name_lower not in seen_names:
            seen_names.add(name_lower)
            unique_companies.append(company)
    
    print(f"Deduplication: {len(companies)} -> {len(unique_companies)} unique companies")
    return unique_companies

def enrich_company_details(company_name: str, search_snippets: List[str]) -> Dict[str, Any]:
    """
    Use LLM to extract and enrich contact details from search result snippets.
    Returns dict with enriched fields: email, phone, address, website, description
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(f"Warning: No OpenAI API Key. Skipping enrichment for {company_name}")
        return {}
    
    client = OpenAI(api_key=api_key)
    
    # Combine snippets into context
    context = "\n\n".join(search_snippets[:20])  # Limit to avoid token overflow
    
    system_prompt = """You are a data enrichment assistant. 
    Your task is to extract and consolidate contact information for a company from search result snippets.
    
    Extract the following if available:
    - email: company email address
    - phone: company phone number
    - address: physical address
    - website: official website URL
    - description: brief company description (2-3 sentences max)
    
    Return JSON:
    {
        "email": "...",
        "phone": "...",
        "address": "...",
        "website": "...",
        "description": "..."
    }
    
    If a field is not found, use null. Prioritize official/primary contact details.
    """
    
    user_prompt = f"""Company Name: {company_name}

Search Results Context:
{context[:10000]}

Extract and return the contact details in JSON format."""
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        enriched_data = json.loads(response.choices[0].message.content)
        return enriched_data
        
    except Exception as e:
        print(f"LLM Enrichment Error for {company_name}: {e}")
        return {}

def enrich_companies(companies: List[Company], country: str = None) -> List[Company]:
    """
    Main enrichment pipeline:
    1. Deduplicate by name
    2. For each unique company, search for additional info (40 results)
    3. Use LLM to enrich contact details
    
    Args:
        companies: List of Company objects to enrich
        country: Optional country name to include in search query for more targeted results
    """
    # Step 1: Deduplicate
    unique_companies = deduplicate_by_name(companies)
    
    # Step 2 & 3: Enrich each company
    enriched_companies = []
    
    for idx, company in enumerate(unique_companies):
        print(f"Enriching {idx+1}/{len(unique_companies)}: {company.name}")
        
        # Search for company contact info (including country if provided)
        if country:
            search_query = f"{company.name} {country} contact information"
        else:
            search_query = f"{company.name} contact information"
        search_results = search_google(search_query, limit=20)
        snippets = [result.get("content", "") for result in search_results if result.get("content")]
        
        if not snippets:
            print(f"  No search results found for {company.name}")
            enriched_companies.append(company)
            continue
        
        # LLM enrichment
        enriched_data = enrich_company_details(company.name, snippets)
        
        # Merge enriched data with existing company data (prefer new data if not empty)
        company.email = enriched_data.get("email") or company.email
        company.phone = enriched_data.get("phone") or company.phone
        company.address = enriched_data.get("address") or company.address
        company.website = enriched_data.get("website") or company.website
        company.description = enriched_data.get("description") or company.description
        
        enriched_companies.append(company)
    
    return enriched_companies
