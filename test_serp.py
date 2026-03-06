import os
from dotenv import load_dotenv
load_dotenv()
from serpapi import GoogleSearch

params = {
    "engine": "amazon_product",
    "amazon_domain": "amazon.com",
    "asin": "B08N5M7S6K",
    "api_key": os.getenv("SERP_API_KEY")
}

search = GoogleSearch(params)
results = search.get_dict()
print(results.keys())
print("product" in results, results.get("product"))
