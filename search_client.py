from duckduckgo_search import DDGS

class DuckDuckGoSearchManager:
    """Manages free web searches using DuckDuckGo to provide live context to agents."""
    
    def execute_search(self, query: str, max_results: int = 5) -> str:
        """Executes a text search and formats the results."""
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                
            if not results:
                return "No search results found."
                
            formatted_results = ""
            for i, r in enumerate(results, 1):
                formatted_results += (
                    f"Result {i}:\n"
                    f"Title: {r.get('title')}\n"
                    f"URL: {r.get('href')}\n"
                    f"Snippet: {r.get('body')}\n"
                    f"{'-'*40}\n"
                )
            return formatted_results
            
        except Exception as e:
            return f"Failed to fetch live data via DuckDuckGo: {str(e)}. Falling back to base knowledge."