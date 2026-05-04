import pathlib, os
base = pathlib.Path(r'C:\Users\sidde.rk\youtube-sentiment')

# ── db.py already written ──────────────────────────────────

# ── youtube_fetcher.py ────────────────────────────────────
(base / 'youtube_fetcher.py').write_text('''
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pytchat, re

def extract_video_id(url: str) -> str:
    for p in [r"youtube.com/watch.v=([\\w-]+)", r"youtu.be/([\\w-]+)", r"youtube.com/shorts/([\\w-]+)"]:
        m = re.search(p, url)
        if m: return m.group(1)
    return url.strip()

def get_video_info(api_key, video_id):
    try:
        yt = build("youtube", "v3", developerKey=api_key)
        resp = yt.videos().list(part="snippet", id=video_id).execute()
        items = resp.get("items", [])
        if not items: return {}
        s = items[0]["snippet"]
        return {"title": s.get("title",""), "channel": s.get("channelTitle",""),
                "description": s.get("description",""), "published": s.get("publishedAt","")}
    except Exception as e:
        return {"error": str(e)}

def get_regular_comments(api_key, video_id, max_results=200):
    yt = build("youtube", "v3", developerKey=api_key)
    comments = []
    try:
        req = yt.commentThreads().list(part="snippet", videoId=video_id,
            maxResults=min(max_results,100), textFormat="plainText", order="relevance")
        while req and len(comments) < max_results:
            resp = req.execute()
            for item in resp.get("items", []):
                sn = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({"text": sn["textDisplay"], "author": sn.get("authorDisplayName",""),
                                  "likes": sn.get("likeCount",0), "source": "comment"})
            req = yt.commentThreads().list_next(req, resp)
    except HttpError as e:
        if "commentsDisabled" in str(e): return []
        raise
    return comments[:max_results]

def get_live_chat(video_id, max_messages=500):
    messages = []
    try:
        chat = pytchat.create(video_id=video_id)
        while chat.is_alive() and len(messages) < max_messages:
            for c in chat.get().sync_items():
                messages.append({"text": c.message, "author": c.author.name,
                                  "likes": 0, "source": "livechat"})
                if len(messages) >= max_messages: break
    except Exception: pass
    return messages
''', encoding='utf-8')

# ── sentiment_analyzer.py ─────────────────────────────────
(base / 'sentiment_analyzer.py').write_text('''
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
_analyzer = SentimentIntensityAnalyzer()

def analyze(items: list) -> list:
    results = []
    for item in items:
        compound = _analyzer.polarity_scores(item["text"])["compound"]
        label = "Positive" if compound >= 0.05 else "Negative" if compound <= -0.05 else "Neutral"
        results.append({**item, "sentiment": label, "score": compound})
    return results

def summary_stats(comments: list) -> dict:
    if not comments: return {}
    from collections import Counter
    counts = Counter(c["sentiment"] for c in comments)
    avg = sum(c["score"] for c in comments) / len(comments)
    return {"total": len(comments), "positive": counts.get("Positive",0),
            "neutral": counts.get("Neutral",0), "negative": counts.get("Negative",0),
            "avg_score": round(avg, 3)}
''', encoding='utf-8')

# ── llm_chat.py ───────────────────────────────────────────
(base / 'llm_chat.py').write_text('''
from openai import OpenAI

def build_context(video_info: dict, comments: list, max_comments: int = 150) -> str:
    desc = (video_info.get("description","") or "")[:1000]
    lines = [
        f"Video Title: {video_info.get(\'title\',\'\')}",
        f"Channel: {video_info.get(\'channel\',\'\')}",
        f"Description: {desc}",
        "",
        f"Total comments analysed: {len(comments)}",
        "",
        "--- Sample Comments (with sentiment score) ---",
    ]
    for c in comments[:max_comments]:
        lines.append(f"[{c[\'source\']}][{c[\'sentiment\']} {c[\'score\']:+.2f}] {c[\'author\']}: {c[\'text\'][:120]}")
    return "\\n".join(lines)

def chat(api_key: str, video_info: dict, comments: list, history: list, user_message: str) -> str:
    client = OpenAI(api_key=api_key)
    context = build_context(video_info, comments)
    system_prompt = f"""You are an AI assistant that analyses YouTube video comments and live chat.
Answer the user\'s questions based ONLY on the data provided below.
If the answer is not in the data, say so clearly.

{context}
"""
    messages = [{"role": "system", "content": system_prompt}]
    for h in history[-10:]:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": user_message})
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages, max_tokens=600)
    return resp.choices[0].message.content
''', encoding='utf-8')

print("All modules written OK")
