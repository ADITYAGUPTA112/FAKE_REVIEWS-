import re

def extract_asin(url_or_asin):
    url_or_asin = url_or_asin.strip()

    # Already ASIN
    if re.fullmatch(r"[A-Z0-9]{10}", url_or_asin):
        return url_or_asin

    # Extract any 10-char ASIN
    match = re.search(r"([A-Z0-9]{10})", url_or_asin)
    if match:
        return match.group(1)

    return None


while True:
    url = input("Paste Amazon URL (or type exit):\n")

    if url.lower() == "exit":
        break

    asin = extract_asin(url)

    if asin:
        print("✅ Extracted ASIN:", asin)
    else:
        print("❌ Could not extract ASIN")