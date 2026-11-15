import json, sys

with open('report.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

last = nb['cells'][-1]
if last['cell_type'] != 'markdown':
    sys.exit("ERROR: last cell is not a markdown cell")

source = ''.join(last['source'])  # source is a list of lines
with open('report.md', 'w', encoding='utf-8') as f:
    f.write(source)
print(f"Wrote report.md ({len(source)} chars) from last cell")