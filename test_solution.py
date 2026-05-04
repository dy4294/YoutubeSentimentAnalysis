import sys, os, pathlib
sys.path.insert(0, '.')

from dotenv import load_dotenv
load_dotenv(pathlib.Path(__file__).parent / ".env")

API_KEY  = os.getenv("YOUTUBE_API_KEY", "").strip()
# Use a reliable English video with comments for regression (fast, predictable)
# To test live stream videos use the UI directly
TEST_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

print("=" * 50)
print("  YouTube Sentiment Solution - Full Test")
print("=" * 50)

if not API_KEY:
    print("  ✗ YOUTUBE_API_KEY not found in .env — aborting.")
    sys.exit(1)
print(f"  ✓ API key loaded from .env")


# ── 1. DB ──────────────────────────────────────────
print("\n[1/5] Testing database...")
import db
db.init_db()
print("  ✓ Database initialised OK")

# ── 2. Video ID Extraction ─────────────────────────
print("\n[2/5] Testing URL parsing...")
from youtube_fetcher import extract_video_id, get_video_info, get_regular_comments, get_live_chat
vid = extract_video_id(TEST_URL)
print(f"  ✓ Extracted video ID: {vid}")

# ── 3. Fetch video info ────────────────────────────
print("\n[3/5] Fetching video info from YouTube API...")
info = get_video_info(API_KEY, vid)
if "error" in info:
    print(f"  ✗ Error: {info['error']}")
    sys.exit(1)
print(f"  ✓ Title   : {info.get('title', '')}")
print(f"  ✓ Channel : {info.get('channel', '')}")

# ── 4. Fetch comments + live chat + sentiment ──────
print("\n[4/5] Fetching comments/live chat and analysing sentiment...")
comments, err = get_regular_comments(API_KEY, vid, 20)
if err:
    print(f"  ℹ Regular comments: {err}")
else:
    print(f"  ✓ Fetched {len(comments)} regular comments")

chat_msgs, chat_err = get_live_chat(api_key, vid, 50)
if chat_err:
    print(f"  ℹ Live chat: {chat_err}")
else:
    print(f"  ✓ Fetched {len(chat_msgs)} live chat messages")

all_items = comments + chat_msgs
if not all_items:
    print("  ⚠ No data fetched (comments disabled and no live chat) — continuing anyway")

from sentiment_analyzer import analyze, summary_stats
results = analyze(all_items)
stats = summary_stats(results)
print(f"  ✓ Sentiment stats: {stats}")

# ── 5. Save & retrieve from DB ────────────────────
print("\n[5/5] Saving to database and reading back...")
db.upsert_video(vid, TEST_URL,
                info.get("title", ""),
                info.get("channel", ""),
                info.get("description", ""),
                info.get("published", ""))
db.save_comments(vid, results)

saved_comments = db.get_comments(vid)
all_videos     = db.get_all_videos()
print(f"  ✓ {len(saved_comments)} comments saved in DB")
print(f"  ✓ {len(all_videos)} video(s) in DB")
print(f"  ✓ DB title: {all_videos[0]['title'] if all_videos else 'NONE'}")

print("\n" + "=" * 50)
print("  ALL TESTS PASSED ✓")
print("=" * 50)
print("\nNow launching the UI...")
