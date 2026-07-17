"""
Quick diagnostic - run this directly with: python check_env.py
It checks whether OPENROUTER_API_KEY is actually being loaded from .env,
without printing the real key (for safety).
"""

import os
from dotenv import load_dotenv

load_dotenv()

key = os.environ.get("OPENROUTER_API_KEY")

if key is None:
    print("PROBLEM: OPENROUTER_API_KEY was not found at all.")
    print("This usually means .env is missing, misnamed, or in the wrong folder.")
elif key.strip() == "":
    print("PROBLEM: OPENROUTER_API_KEY exists but is EMPTY.")
    print("Check that there's a real key after the = sign in .env, with no line break.")
else:
    masked = key[:8] + "..." + key[-4:] if len(key) > 12 else "(too short to mask safely)"
    print(f"OK: Key was loaded. Starts/ends with: {masked}")
    print(f"Key length: {len(key)} characters")
    if key.startswith('"') or key.endswith('"') or key.startswith("'"):
        print("WARNING: Key appears to have quote characters in it - remove any quotes from .env")
    if " " in key:
        print("WARNING: Key contains a space - check for accidental extra spaces")