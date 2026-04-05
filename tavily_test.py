import json
from tavily import TavilyClient
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize the client with your API key
client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

# --- Option 1: Search ---
search_results = client.search(
    query="What is artificial intelligence?",
    search_depth="advanced",       # "basic" or "advanced"
    include_raw_content=True,      # include full page content
    max_results=3,
)

print("=== SEARCH RESULTS ===")

urls = [r["url"] for r in search_results.get("results", [])[:3]]
if urls:
    try:
        extracted = client.extract(urls=urls)
        search_results["extracted_content"] = extracted
        print(extracted)
    except Exception:
        pass



#print(json.dumps(search_results, indent=2, ensure_ascii=False))

# Print each result individually
# for i, result in enumerate(search_results["results"]):
#     print(f"\n--- Result {i+1} ---")
#     print(f"Title:   {result['title']}")
#     print(f"URL:     {result['url']}")
#     print(f"Score:   {result['score']}")
#     print(f"Snippet: {result['content'][:300]}...")
#     if result.get("raw_content"):
#         print(f"Raw content (first 500 chars): {result['raw_content'][:500]}...")


# --- Option 2: Extract (fetch a specific URL) ---
# extract_results = client.extract(
#     urls=["https://en.wikipedia.org/wiki/Artificial_intelligence"]
# )

# print("\n=== EXTRACT RESULTS ===")
# print(json.dumps(extract_results, indent=2, ensure_ascii=False))

# for item in extract_results["results"]:
#     print(f"\nURL: {item['url']}")
#     print(f"Content (first 1000 chars):\n{item['raw_content'][:1000]}...")