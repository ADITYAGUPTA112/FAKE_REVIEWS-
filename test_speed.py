import time
from analyzer import analyze_product

start_time = time.time()
print("Starting analysis...")
# Providing an ASIN to test
summary, df = analyze_product("B08N5M7S6K", pages=1) # 1 page is fast enough to test
end_time = time.time()
print(f"Analysis took {end_time - start_time:.2f} seconds")
print(summary)
