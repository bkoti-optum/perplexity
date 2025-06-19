"""Tools module for external services."""

from typing import Dict, Any
import requests

class TavilySearchResults:
    """Tavily search tool implementation."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.tavily.com/v1/search"
    
    def invoke(self, query: Dict[str, str]) -> list[Dict[str, Any]]:
        """Perform a search using Tavily API."""
        headers = {"X-Api-Key": self.api_key}
        response = requests.post(
            self.base_url,
            json={"query": query["query"]},
            headers=headers
        )
        return response.json().get("results", []) 