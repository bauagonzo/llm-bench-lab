import os
import re

imports = set()

for filename in os.listdir('tools'):
    if filename.endswith('.py'):
        with open(f'tools/{filename}', 'r') as f:
            for line in f:
                if line.strip().startswith('import ') or line.strip().startswith('from '):
                    # Extract just the package name
                    match = re.match(r'(?:import|from)\s+(\w+)', line)
                    if match:
                        imports.add(match.group(1))

for pkg in sorted(imports):
    print(pkg)