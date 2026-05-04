"""
Full mojibake recovery for app.py.
Strategy: treat the entire file's non-ASCII chars as cp1252 -> re-encode to bytes -> decode as UTF-8.
"""
import pathlib

f = pathlib.Path(__file__).parent / "app.py"
raw = f.read_text(encoding="utf-8")

# Encode the whole string char-by-char back to cp1252 bytes, then decode those bytes as UTF-8.
# This reverses: original_utf8_bytes -> decoded_as_cp1252 -> encoded_as_utf8
try:
    recovered_bytes = raw.encode("cp1252")
    fixed = recovered_bytes.decode("utf-8")
    print("Full cp1252->utf8 round-trip worked")
except Exception as e:
    print(f"Full round-trip failed: {e}")
    # Fallback: char-by-char, skip chars that can't encode to cp1252
    buf = bytearray()
    for ch in raw:
        try:
            buf += ch.encode("cp1252")
        except UnicodeEncodeError:
            # Already a proper Unicode char that wasn't mojibaked — encode as UTF-8 placeholder
            buf += ch.encode("utf-8")
    # Now decode as UTF-8 with error replacement
    fixed = buf.decode("utf-8", errors="replace")
    print("Used char-by-char fallback")

import ast
try:
    ast.parse(fixed)
    print("AST parse OK")
except SyntaxError as e:
    print(f"SyntaxError: {e}")
    exit(1)

# Spot check
lines = fixed.splitlines()
for i in [23, 78, 103, 113, 167]:
    if i < len(lines):
        print(f"  line {i+1}: {lines[i][:90]}")

remaining = sum(1 for c in fixed if "\u0080" <= c <= "\u02ff")
print(f"Remaining suspicious chars: {remaining}")

f.write_text(fixed, encoding="utf-8")
print("Saved.")
