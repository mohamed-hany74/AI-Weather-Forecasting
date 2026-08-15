"""
Security Scanner for WeatherCast AI repository.
Scans for hardcoded secrets, API keys, tokens, machine paths, and credentials.
"""

import os
import re

PATTERNS = [
    (re.compile(r'sk-[a-zA-Z0-9]{20,}'), 'OpenAI API Key'),
    (re.compile(r'ghp_[a-zA-Z0-9]{30,}'), 'GitHub Personal Access Token'),
    (re.compile(r'AIza[0-9A-Za-z\-_]{35}'), 'Google/Gemini API Key'),
    (re.compile(r'AKIA[0-9A-Z]{16}'), 'AWS Access Key ID'),
    (re.compile(r'django-insecure-[a-zA-Z0-9_\-\.\+=\!@#\$%\^&\*]{10,}', re.I), 'Django Insecure Secret Key'),
    (re.compile(r'-----BEGIN (RSA|EC|DSA|OPENSSH) PRIVATE KEY-----'), 'Private SSH/RSA Key'),
    (re.compile(r'(?i)(password|secret_key|api_key|access_token|bearer)\s*=\s*[\'"][^\'"]{8,}[\'"]'), 'Hardcoded Secret Assignment'),
    (re.compile(r'[a-zA-Z]:\\[Uu]sers\\[a-zA-Z0-9_\-\.]+'), 'Hardcoded Windows User Path')
]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
findings = []

EXCLUDED_DIRS = {'.git', '.venv', '__pycache__', '_backup', '.idea', '.vscode'}
EXCLUDED_EXTS = {'.pyc', '.sqlite3', '.png', '.jpg', '.jpeg', '.gif', '.ico', '.pkl', '.joblib', '.keras', '.tar', '.zip'}

for root, dirs, files in os.walk(PROJECT_ROOT):
    dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
    for f in files:
        if any(f.endswith(ext) for ext in EXCLUDED_EXTS):
            continue
        filepath = os.path.join(root, f)
        rel_path = os.path.relpath(filepath, PROJECT_ROOT)
        
        # Don't scan the scanner script itself or example files
        if rel_path == 'scripts/security_scan.py' or 'example' in rel_path.lower():
            continue
            
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as fp:
                for lineno, line in enumerate(fp, start=1):
                    for regex, desc in PATTERNS:
                        m = regex.search(line)
                        if m:
                            matched_str = m.group(0)
                            # Filter safe patterns (e.g. os.environ.get, docstrings, example secrets in docker-compose example)
                            if 'change-this' in matched_str or 'your-production' in matched_str or 'os.environ' in line:
                                continue
                            masked = matched_str[:6] + '***' + matched_str[-4:] if len(matched_str) > 12 else '***'
                            findings.append((rel_path, lineno, desc, masked, line.strip()[:100]))
        except Exception as e:
            pass

print("=" * 80)
print(f"       SECURITY SCAN RESULTS: {len(findings)} FINDINGS DETECTED")
print("=" * 80)

for path, lineno, desc, masked, snippet in findings:
    print(f"[{desc}] {path}:{lineno}")
    print(f"  Snippet: {snippet}")

if not findings:
    print("[PASS] Zero secrets or sensitive machine information detected in repository.")
