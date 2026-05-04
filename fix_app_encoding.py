"""
Fix cp1252 mojibake in app.py.
What happened:
  original UTF-8 bytes  →  decoded as cp1252  →  re-encoded as UTF-8  →  stored in file
Reverse per sequence:
  read as UTF-8  →  encode back to cp1252 bytes  →  decode those bytes as UTF-8  →  original chars
"""
import re, pathlib

f = pathlib.Path(__file__).parent / "app.py"
raw = f.read_text(encoding="utf-8")

def try_fix_chunk(m):
    chunk = m.group(0)
    try:
        return chunk.encode("cp1252").decode("utf-8")
    except Exception:
        return chunk  # leave unchanged if it doesn't fix cleanly

# Match runs of chars in the Windows-1252 extended range that got mojibaked
# These are chars U+0080-U+017F which cp1252 uses for its upper half
fixed = re.sub(r"[\u0080-\u017f]+", try_fix_chunk, raw)

import ast
try:
    ast.parse(fixed)
    print("AST parse OK")
except SyntaxError as e:
    print(f"SyntaxError: {e}"); exit(1)

line24 = fixed.splitlines()[23]
print(f"Line 24: {line24}")
print(f"Emoji 🎬 fixed: {chr(0x1F3AC) in line24}")

f.write_text(fixed, encoding="utf-8")
print("Done — saved clean UTF-8 app.py")


