import sys
for file in sys.argv[1:]:
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
    idx = content.rfind('<g><text style="font-size: 32px; font-weight: bold;"')
    if idx != -1:
        content = content[:idx] + '</svg>'
        with open(file, 'w', encoding='utf-8') as f:
            f.write(content)
