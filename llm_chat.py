
from openai import OpenAI

def build_context(video_info: dict, comments: list, max_comments: int = 150) -> str:
    desc = (video_info.get("description","") or "")[:1000]
    vc  = int(video_info.get("view_count",  0) or 0)
    lc  = int(video_info.get("like_count",  0) or 0)
    cc  = int(video_info.get("comment_count", 0) or 0)
    lines = [
        f"Video Title: {video_info.get('title','')}",
        f"Channel: {video_info.get('channel','')}",
        f"Description: {desc}",
        f"Views: {vc:,}",
        f"Likes: {lc:,}",
        f"YouTube Comment Count: {cc:,}  (Note: YouTube removed public dislike counts in Dec 2021)",
        "",
        f"Total comments analysed: {len(comments)}",
        "",
        "--- Sample Comments (with sentiment score) ---",
    ]
    for c in comments[:max_comments]:
        lines.append(f"[{c['source']}][{c['sentiment']} {c['score']:+.2f}] {c['author']}: {c['text'][:120]}")
    return "\n".join(lines)

def chat(api_key: str, video_info: dict, comments: list, history: list, user_message: str) -> str:
    client = OpenAI(api_key=api_key)
    context = build_context(video_info, comments)
    system_prompt = f"""You are an AI assistant that analyses YouTube video comments and live chat.
Answer the user's questions based ONLY on the data provided below.
If the answer is not in the data, say so clearly.

IMPORTANT — Comments to IGNORE during analysis:
- Short hailing / chanting comments such as "Jai Vijay", "Jai Hind", "Zindabad", "Viva", "Hurray", "Bravo", or any variant in any language.
- These are crowd cheers with no analytical meaning. Do NOT count them as Positive sentiment, do NOT cite them as evidence of support, and do NOT include them in theme or topic analysis.
- Only analyse comments that express a real opinion, emotion, question, or observation.

{context}
"""
    messages = [{"role": "system", "content": system_prompt}]
    for h in history[-10:]:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": user_message})
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages, max_tokens=600)
    return resp.choices[0].message.content
