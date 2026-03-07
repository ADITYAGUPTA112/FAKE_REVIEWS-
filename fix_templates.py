import os
import re

folder = os.path.join(os.path.dirname(__file__), 'templates')

for fname in os.listdir(folder):
    if not fname.endswith('.html'):
        continue
    fpath = os.path.join(folder, fname)
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Fix backslash-escaped quotes inside url_for like url_for(\'index\')  ->  url_for('index')
    content = content.replace("\\'", "'")

    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Fixed: {fname}")

print("Done!")
