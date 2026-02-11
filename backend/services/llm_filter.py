from typing import List, Dict
import os
import json

# Try to import openai, but handle if it's not present (though we should probably add it to requirements)
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

def filter_search_results(results: List[Dict[str, str]], query: str) -> List[str]:
    """
    Filters search results based on the query using an LLM.
    Returns a list of URLs that are relevant.
    """
    if not results:
        return []

    # If OpenAI is not available or no key, return all results (fail open)
    # This is a placeholder. ideally we should have a free LLM or user provided key.
    # The user mentioned "Crawl4AI with its AI capabilities", but that's for extraction.
    # For now, let's use a dummy filter if no key, or try to use a free provider if possible.
    # Actually, let's simply return all results if we can't filter, but print a warning.
    
    # Check for API Key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not OpenAI or not api_key:
        print("Warning: OpenAI client not available or API key missing. Skipping LLM filtering.")
        return [r["url"] for r in results]

    client = OpenAI(api_key=api_key)
    
    # Prepare the prompt
    candidates = ""
    for i, r in enumerate(results):
        candidates += f"{i}. URL: {r['url']}\n   Snippet: {r['content'][:200]}\n"

    system_prompt = """
    You are a strict search relevance evaluator.

    Your task is to determine whether each snippet is relevant to the user's query.

    Definition of Relevant:
    A snippet is relevant ONLY if it directly helps answer, explain, or satisfy the user’s search intent.

    Relevance Rules:
    1. The snippet must directly address the core intent of the query.
    2. Superficial keyword overlap is NOT sufficient.
    3. Broad topical similarity is NOT sufficient.
    4. The snippet must contain specific information related to the query.
    5. If the snippet would not likely be clicked by a user searching this query, ignore it.
    Analyze the snippets and return a JSON list of indices of the relevant results.
    Example output: [0, 2, 4]
    Only return the JSON list, nothing else."""

    user_prompt = f"User Query: {query}\n\nSearch Results:\n{candidates}\n\nWhich of these are relevant to the query?"

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", # Or gpt-4o-mini
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse output
        try:
            indices = json.loads(content)
            if isinstance(indices, list):
                valid_urls = []
                for idx in indices:
                    if isinstance(idx, int) and 0 <= idx < len(results):
                        valid_urls.append(results[idx]["url"])
                print(f"LLM Filtered {len(results)} -> {len(valid_urls)} URLs")
                return valid_urls
        except json.JSONDecodeError:
            print(f"Failed to parse LLM response: {content}")
            return [r["url"] for r in results] # Fail open

    except Exception as e:
        print(f"Error calling LLM: {e}")
        return [r["url"] for r in results] # Fail open

    return [r["url"] for r in results]
