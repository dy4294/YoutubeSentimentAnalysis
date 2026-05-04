
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import streamlit as st
import pandas as pd
import plotly.express as px
from db import (get_all_videos, get_video, get_comments, save_comments,
                upsert_video, save_chat_message, get_chat_history, clear_chat_history)
from youtube_fetcher import extract_video_id, get_video_info, get_regular_comments, get_live_chat
from sentiment_analyzer import analyze, summary_stats
from llm_chat import chat as llm_chat

st.set_page_config(page_title="YouTube Sentiment Analyser", page_icon="🎬", layout="wide")

# ── persist API keys across reruns via session_state ─────────────────────────
if "yt_key"  not in st.session_state: st.session_state["yt_key"]  = ""
if "oai_key" not in st.session_state: st.session_state["oai_key"] = ""

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")

    yt_key = st.text_input("YouTube API Key", type="password",
                           value=st.session_state["yt_key"],
                           help="Get from console.cloud.google.com")
    if yt_key:
        st.session_state["yt_key"] = yt_key

    oai_key = st.text_input("OpenAI API Key (for AI Chat)", type="password",
                            value=st.session_state["oai_key"],
                            help="Get from platform.openai.com — optional")
    if oai_key:
        st.session_state["oai_key"] = oai_key

    st.markdown("---")
    st.subheader("📂 Previously Analysed")
    all_videos = get_all_videos()
    if all_videos:
        for v in all_videos:
            label = (v["title"] or v["video_id"])[:38]
            if st.button(f"🎬 {label}", key=f"v_{v['video_id']}", use_container_width=True):
                st.session_state["selected_video_id"] = v["video_id"]
                st.rerun()
    else:
        st.caption("No videos yet.")

# ── HEADER ────────────────────────────────────────────────────────────────────
st.title("🎬 YouTube Sentiment Analyser")
st.markdown("Paste a YouTube URL → fetch comments & live chat → sentiment analysis → AI Q&A.")

# ── INPUT ROW ────────────────────────────────────────────────────────────────
url_input   = st.text_input("🔗 YouTube Video URL",
                             placeholder="https://www.youtube.com/watch?v=...")
col1, col2  = st.columns([1, 4])
with col1:
    analyse_btn = st.button("🔍 Analyse", type="primary", use_container_width=True)
with col2:
    max_comments = st.slider("Max comments to fetch", 50, 500, 200, 50)

# ── ANALYSE BUTTON LOGIC ─────────────────────────────────────────────────────
if analyse_btn:
    active_key = st.session_state.get("yt_key", "").strip()
    if not active_key:
        st.error("❌ Enter your YouTube API Key in the sidebar first.")
        st.stop()
    if not url_input.strip():
        st.error("❌ Enter a YouTube URL.")
        st.stop()

    video_id = extract_video_id(url_input.strip())
    st.session_state["selected_video_id"] = video_id

    progress = st.progress(0, text="Fetching video info...")

    info = get_video_info(active_key, video_id)
    if not info or "error" in info:
        st.error(f"❌ Could not fetch video: {info.get('error', 'Video not found. Check the URL.')}")
        st.stop()

    upsert_video(video_id, url_input.strip(),
                 info.get("title",""), info.get("channel",""),
                 info.get("description",""), info.get("published",""))
    progress.progress(20, text=f"Got video: {info.get('title','')[:60]}")

    progress.progress(30, text="Fetching regular comments...")
    raw_comments = get_regular_comments(active_key, video_id, max_comments)
    progress.progress(60, text=f"Got {len(raw_comments)} comments. Fetching live chat...")

    raw_chat = get_live_chat(video_id, 500)
    progress.progress(80, text=f"Got {len(raw_chat)} live chat messages. Running sentiment analysis...")

    all_raw = analyze(raw_comments + raw_chat)
    save_comments(video_id, all_raw)
    progress.progress(100, text="Done!")

    st.success(
        f"✅ **Analysis complete!** "
        f"{len(raw_comments)} comments + {len(raw_chat)} live chat messages analysed."
    )

# ── DISPLAY SELECTED VIDEO ────────────────────────────────────────────────────
selected_id = st.session_state.get("selected_video_id")

if not selected_id:
    st.info("👆 Enter a YouTube URL above and click **Analyse** to get started.")
    st.stop()

video    = get_video(selected_id)
all_data = get_comments(selected_id)

if not video:
    st.warning("Video not found in database. Click Analyse to fetch it.")
    st.stop()

if not all_data:
    st.warning("No comments in database yet. Click **Analyse** to fetch them.")
    st.stop()

df          = pd.DataFrame(all_data)
df_comments = df[df["source"] == "comment"].reset_index(drop=True)
df_chat     = df[df["source"] == "livechat"].reset_index(drop=True)

# ── VIDEO HEADER ─────────────────────────────────────────────────────────────
st.markdown(f"## 📺 {video.get('title','')}")
st.caption(f"Channel: **{video.get('channel','')}**  |  Fetched: {video.get('fetched_at','')[:16]} UTC")
with st.expander("📄 Video Description"):
    st.write(video.get("description","No description available."))

tab1, tab2, tab3 = st.tabs(["📊 Sentiment Overview", "💬 Comments Browser", "🤖 AI Chat"])

# ── TAB 1: SENTIMENT OVERVIEW ─────────────────────────────────────────────────
with tab1:
    c1, c2 = st.columns(2)
    color_map = {"Positive": "#2ecc71", "Neutral": "#95a5a6", "Negative": "#e74c3c"}

    for col, df_src, label, src_key in [
        (c1, df_comments, "💬 Regular Comments", "comment"),
        (c2, df_chat,     "🔴 Live Chat",        "livechat"),
    ]:
        with col:
            st.markdown(f"#### {label}")
            if df_src.empty:
                if src_key == "comment":
                    st.warning("Comments are disabled for this video.")
                else:
                    st.info("No live chat found (not a live stream).")
            else:
                stats  = summary_stats(df_src.to_dict("records"))
                m1,m2,m3,m4 = st.columns(4)
                m1.metric("Total",    stats["total"])
                m2.metric("✅ Positive", stats["positive"])
                m3.metric("⚪ Neutral",  stats["neutral"])
                m4.metric("❌ Negative", stats["negative"])
                st.caption(f"Average sentiment score: **{stats['avg_score']:+.3f}**")

                counts = df_src["sentiment"].value_counts().reset_index()
                counts.columns = ["Sentiment","Count"]
                fig = px.bar(counts, x="Sentiment", y="Count", color="Sentiment",
                             color_discrete_map=color_map, text="Count",
                             title=f"{label} Distribution")
                fig.update_layout(showlegend=False, height=300, margin=dict(t=40,b=0))
                st.plotly_chart(fig, use_container_width=True)

    # Overall pie
    if not df.empty:
        st.markdown("---")
        pc1, pc2 = st.columns(2)
        with pc1:
            st.markdown("#### 🥧 Overall Sentiment (all sources)")
            overall = df["sentiment"].value_counts().reset_index()
            overall.columns = ["Sentiment","Count"]
            fig2 = px.pie(overall, names="Sentiment", values="Count",
                          color="Sentiment", color_discrete_map=color_map, hole=0.4)
            fig2.update_layout(height=320)
            st.plotly_chart(fig2, use_container_width=True)
        with pc2:
            st.markdown("#### 📈 Score Distribution")
            fig3 = px.histogram(df, x="score", color="source", nbins=40,
                                barmode="overlay", opacity=0.75,
                                color_discrete_map={"comment":"#3498db","livechat":"#e67e22"},
                                labels={"score":"VADER Score","source":"Source"})
            fig3.add_vline(x=0.05,  line_dash="dash", line_color="green")
            fig3.add_vline(x=-0.05, line_dash="dash", line_color="red")
            fig3.update_layout(height=320)
            st.plotly_chart(fig3, use_container_width=True)

    # Top comments
    if not df.empty:
        st.markdown("---")
        tp1, tp2 = st.columns(2)
        with tp1:
            st.markdown("#### 🟢 Top 5 Most Positive")
            for _, r in df.nlargest(5,"score").iterrows():
                st.markdown(f"`{r['score']:+.2f}` [{r['source']}] **{r['author'][:20]}**: {r['text'][:90]}")
        with tp2:
            st.markdown("#### 🔴 Top 5 Most Negative")
            for _, r in df.nsmallest(5,"score").iterrows():
                st.markdown(f"`{r['score']:+.2f}` [{r['source']}] **{r['author'][:20]}**: {r['text'][:90]}")

# ── TAB 2: COMMENTS BROWSER ───────────────────────────────────────────────────
with tab2:
    f1, f2, f3 = st.columns(3)
    src_filter  = f1.selectbox("Source",    ["All","comment","livechat"])
    sent_filter = f2.selectbox("Sentiment", ["All","Positive","Neutral","Negative"])
    search_term = f3.text_input("🔎 Search text")

    filtered = df.copy()
    if src_filter  != "All": filtered = filtered[filtered["source"]   == src_filter]
    if sent_filter != "All": filtered = filtered[filtered["sentiment"] == sent_filter]
    if search_term:
        filtered = filtered[filtered["text"].str.contains(search_term, case=False, na=False)]

    st.caption(f"Showing **{len(filtered)}** of **{len(df)}** records")
    st.dataframe(
        filtered[["source","sentiment","score","author","text","likes"]].sort_values("score", ascending=False),
        use_container_width=True, height=450,
        column_config={
            "score":     st.column_config.NumberColumn("Score",   format="%.3f"),
            "sentiment": st.column_config.TextColumn("Sentiment"),
            "source":    st.column_config.TextColumn("Source"),
            "text":      st.column_config.TextColumn("Comment",  width="large"),
            "likes":     st.column_config.NumberColumn("Likes"),
        }
    )
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, "comments.csv", "text/csv")

# ── TAB 3: AI CHAT ────────────────────────────────────────────────────────────
with tab3:
    st.subheader("🤖 Ask AI about this Video")
    active_oai = st.session_state.get("oai_key","").strip()

    if not active_oai:
        st.info(
            "Enter your **OpenAI API Key** in the sidebar to enable AI Chat.  \n"
            "Get a key at [platform.openai.com](https://platform.openai.com) — "
            "free tier available with $5 credit."
        )
    else:
        history = get_chat_history(selected_id)

        # Render history
        for msg in history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        col_clr, _ = st.columns([1,5])
        if history and col_clr.button("🗑️ Clear History"):
            clear_chat_history(selected_id)
            st.rerun()

        user_msg = st.chat_input("Ask anything about this video's comments or sentiment...")
        if user_msg:
            with st.chat_message("user"):
                st.write(user_msg)
            save_chat_message(selected_id, "user", user_msg)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        reply = llm_chat(
                            api_key=active_oai,
                            video_info=video,
                            comments=all_data,
                            history=history,
                            user_message=user_msg,
                        )
                        st.write(reply)
                        save_chat_message(selected_id, "assistant", reply)
                    except Exception as e:
                        st.error(f"LLM error: {e}")
            st.rerun()
