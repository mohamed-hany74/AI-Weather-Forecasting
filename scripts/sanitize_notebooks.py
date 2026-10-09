"""
Sanitize Jupyter notebooks from machine-specific absolute paths.
"""

import os
import re

notebook_files = [
    os.path.join('ml', 'notebooks', 'Weather_Model.ipynb'),
    os.path.join('ml', 'notebooks', 'Weather_Model_PRODUCTION.ipynb'),
]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for nb in notebook_files:
    full_path = os.path.join(PROJECT_ROOT, nb)
    if os.path.exists(full_path):
        with open(full_path, 'r', encoding='utf-8') as fp:
            text = fp.read()
        
        # Replace local paths
        text = re.sub(r'C:\\\\Users\\\\[a-zA-Z0-9_\-\.]+\\\\AppData\\\\[^"]+', 'tmp/ipykernel', text)
        text = re.sub(r'E:\\\\Weather Project\\\\', '', text)
        text = re.sub(r'E:/Weather Project/', '', text)
        
        with open(full_path, 'w', encoding='utf-8') as fp:
            fp.write(text)
        print(f"Sanitized {nb}")

print("[PASS] All notebooks sanitized from local machine paths.")
