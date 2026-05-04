from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pytchat
import pandas as pd
import matplotlib.pyplot as plt
import os

# CONFIG
API_KEY      = "AIzaSyBeA48567eMU7gAm0uigblAiopIO1JJxaM"
VIDEO_ID     = "xCydcAzdYr8"
MAX_COMMENTS = 200
MAX_CHAT     = 500


def get_regular_comments(video_id, max_results=200):
    youtube = build("youtube", "v3", developerKey=API_KEY)
    comments = []
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(max_results, 100),
            textFormat="plainText",
            order="relevance"
        )
        while request and len(comments) < max_results:
            response = request.execute()
            for item in response.get("items", []):
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "text": snippet["textDisplay"],
                    "author": snippet.get("authorDisplayName", ""),
                    "likes": snippet.get("likeCount", 0),
                    "source": "comment"
                })
            request = youtube.commentThreads().list_next(request, response)
        return comments[:max_results]
    except HttpError as e:
        reason = str(e)
        if "commentsDisabled" in reason:
            print("  [!] Regular comments are DISABLED for this video.")
        else:
            print(f"  [!] Error fetching comments: {e}")
        return []


def get_live_chat(video_id, max_messages=500):
    messages = []
    try:
        chat = pytchat.create(video_id=video_id)
        count = 0
        print(f"  Fetching live chat (max {max_messages} messages)...")
        while chat.is_alive() and count < max_messages:
            for c in chat.get().sync_items():
                messages.append({
                    "text": c.message,
                    "author": c.author.name,
                    "likes": 0,
                    "source": "livechat"
                })
                count += 1
                if count >= max_messages:
                    break
        if not messages:
            print("  [!] No live chat found. This may not be a live stream video.")
    except Exception as e:
        print(f"  [!] Error fetching live chat: {e}")
    return messages


def analyze_sentiment(items):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    results = []
    for item in items:
        text = item["text"]
        compound = analyzer.polarity_scores(text)["compound"]
        if compound >= 0.05:
            label = "Positive"
        elif compound <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"
        results.append({
            "source":    item["source"],
            "author":    item["author"],
            "comment":   text,
            "sentiment": label,
            "score":     compound,
            "likes":     item["likes"]
        })
    return results


def print_summary(df, label):
    print(f"\n{'='*45}")
    print(f"  {label}")
    print(f"{'='*45}")
    if df.empty:
        print("  No data available.")
        return
    print(df["sentiment"].value_counts().to_string())
    print(f"\n  Total       : {len(df)}")
    print(f"  Avg score   : {df['score'].mean():.3f}")
    print("\n  -- Top 3 Most Positive --")
    for _, row in df.nlargest(3, "score").iterrows():
        print(f"    [{row['score']:+.2f}] {row['comment'][:75]}")
    print("\n  -- Top 3 Most Negative --")
    for _, row in df.nsmallest(3, "score").iterrows():
        print(f"    [{row['score']:+.2f}] {row['comment'][:75]}")


def plot_chart(df_comments, df_chat):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"YouTube Sentiment Analysis  |  Video: {VIDEO_ID}", fontsize=13)
    color_map = {"Positive": "green", "Neutral": "gray", "Negative": "red"}
    for ax, df, title in [(axes[0], df_comments, "Regular Comments"), (axes[1], df_chat, "Live Chat Messages")]:
        if df.empty:
            ax.set_title(f"{title}\n(No data)")
            ax.axis("off")
        else:
            counts = df["sentiment"].value_counts()
            colors = [color_map.get(s, "blue") for s in counts.index]
            counts.plot(kind="bar", ax=ax, color=colors, edgecolor="black")
            ax.set_title(title)
            ax.set_xlabel("Sentiment")
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=0)
            for p in ax.patches:
                ax.annotate(str(int(p.get_height())), (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="bottom", fontsize=11)
    plt.tight_layout()
    chart_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sentiment_chart.png")
    plt.savefig(chart_path)
    plt.show()
    print(f"\nChart saved to: {chart_path}")


if __name__ == "__main__":
    print(f"\nAnalyzing video: https://www.youtube.com/watch?v={VIDEO_ID}\n")
    print("[1/2] Fetching regular comments...")
    raw_comments = get_regular_comments(VIDEO_ID, MAX_COMMENTS)
    print(f"  Found {len(raw_comments)} comments.")
    print("\n[2/2] Fetching live chat messages...")
    raw_chat = get_live_chat(VIDEO_ID, MAX_CHAT)
    print(f"  Found {len(raw_chat)} chat messages.")
    all_results = analyze_sentiment(raw_comments + raw_chat)
    df_all = pd.DataFrame(all_results)
    df_comments = df_all[df_all["source"] == "comment"].reset_index(drop=True)
    df_chat     = df_all[df_all["source"] == "livechat"].reset_index(drop=True)
    print_summary(df_comments, "REGULAR COMMENTS SENTIMENT")
    print_summary(df_chat,     "LIVE CHAT SENTIMENT")
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.csv")
    df_all.to_csv(csv_path, index=False)
    print(f"\nAll results saved to: {csv_path}")
    plot_chart(df_comments, df_chat)

