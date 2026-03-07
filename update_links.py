import os
import re

routes = {
    'Home': 'index',
    'Analyze': 'analyze',
    'History': 'history',
    'Leaderboard': 'leaderboard',
    'Profile': 'profile',
    'Login here': 'login',
    'Login': 'login',
    'Register now': 'register'
}

folder = os.path.abspath('templates')

for fname in os.listdir(folder):
    if not fname.endswith('.html'):
        continue
    filepath = os.path.join(folder, fname)
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Generic simple link matching for exact text like ">Home</a>"
    for text, route in routes.items():
        pattern = r'href="#"([^>]*)>\s*' + re.escape(text) + r'\s*<'
        repl = r'href="{{ url_for(\'' + route + r'\') }}"\1>' + text + r'<'
        content = re.sub(pattern, repl, content)

    # Sidebar links with spans (e.g. <span class="material-symbols-outlined">home</span> Home)
    for text, route in routes.items():
        pattern = r'href="#"([^>]*)>([\s\n]*<span[^>]*>[^<]*</span>\s*)' + re.escape(text) + r'(\s*</a>)'
        repl = r'href="{{ url_for(\'' + route + r'\') }}"\1>\2' + text + r'\3'
        content = re.sub(pattern, repl, content)

    # Logout buttons with onclick
    content = re.sub(
        r'(<button[^>]*?)(\s*>\s*<span[^>]*>\s*Logout\s*</span>\s*</button>)',
        r'\1 onclick="window.location.href=\'{{ url_for(\'logout\') }}\'"\2',
        content,
        flags=re.IGNORECASE
    )
    content = re.sub(
        r'(<button[^>]*?)(\s*>\s*Logout\s*</button>)',
        r'\1 onclick="window.location.href=\'{{ url_for(\'logout\') }}\'"\2',
        content,
        flags=re.IGNORECASE
    )
    
    # Extra specifically for Logout buttons that have span icon + text side by side
    content = re.sub(
        r'(<button[^>]*?)(\s*>\s*<span[^>]*>logout</span>[\s\n]*Logout[\s\n]*</button>)',
        r'\1 onclick="window.location.href=\'{{ url_for(\'logout\') }}\'"\2',
        content,
        flags=re.IGNORECASE
    )

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

print("Links updated successfully!")
