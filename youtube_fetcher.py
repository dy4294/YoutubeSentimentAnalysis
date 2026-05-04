
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pytchat, re

def extract_video_id(url: str) -> str:
    for p in [r"youtube\.com/watch\?v=([\w-]+)", r"youtu\.be/([\w-]+)", r"youtube\.com/shorts/([\w-]+)"]:
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
        err = str(e)
        if "commentsDisabled" in err:
            return [], "Comments are disabled for this video."
        if "keyInvalid" in err or "badRequest" in err or "API key not valid" in err:
            return [], f"Invalid API key: {err}"
        return [], f"YouTube API error: {err}"
    except Exception as e:
        return [], f"Unexpected error: {e}"
    return comments[:max_results], None

def get_live_chat(api_key, video_id, max_messages=500):
    """Fetch live chat messages using the YouTube Data API (primary) with pytchat as fallback."""
    messages = []

    # --- Primary: YouTube Data API v3 liveChatMessages ---
    try:
        yt = build("youtube", "v3", developerKey=api_key)
        # Get the activeLiveChatId from the video details
        vid_resp = yt.videos().list(
            part="liveStreamingDetails,snippet", id=video_id
        ).execute()
        items = vid_resp.get("items", [])
        if items:
            live_details = items[0].get("liveStreamingDetails", {})
            live_chat_id = live_details.get("activeLiveChatId")
            if live_chat_id:
                page_token = None
                while len(messages) < max_messages:
                    kwargs = dict(liveChatId=live_chat_id,
                                  part="snippet,authorDetails",
                                  maxResults=min(200, max_messages - len(messages)))
                    if page_token:
                        kwargs["pageToken"] = page_token
                    resp = yt.liveChatMessages().list(**kwargs).execute()
                    for item in resp.get("items", []):
                        text = item["snippet"].get("displayMessage", "")
                        author = item["authorDetails"].get("displayName", "")
                        if text:
                            messages.append({"text": text, "author": author,
                                             "likes": 0, "source": "livechat"})
                    page_token = resp.get("nextPageToken")
                    if not page_token or len(messages) >= max_messages:
                        break
                if messages:
                    return messages[:max_messages], None
    except HttpError as e:
        err = str(e)
        if "forbidden" in err.lower() or "livechat" in err.lower():
            pass  # fall through to pytchat
        else:
            return [], f"Live chat API error: {err}"
    except Exception:
        pass  # fall through to pytchat

    # --- Fallback: pytchat (works for some archived streams) ---
    try:
        import time
        chat = pytchat.create(video_id=video_id)
        deadline = time.time() + 8  # max 8 seconds
        while chat.is_alive() and len(messages) < max_messages and time.time() < deadline:
            for c in chat.get().sync_items():
                messages.append({"text": c.message, "author": c.author.name,
                                  "likes": 0, "source": "livechat"})
                if len(messages) >= max_messages:
                    break
    except Exception as e:
        if not messages:
            return [], f"Live chat unavailable: {e}"

    return messages[:max_messages], (None if messages else "No live chat messages found.")
