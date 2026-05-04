import sys, pathlib, os, re
from html import escape as h
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(pathlib.Path(__file__).parent / ".env")

YT_KEY  = os.getenv("YOUTUBE_API_KEY", "").strip()
OAI_KEY = os.getenv("OPENAI_API_KEY",  "").strip()

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from db import (get_all_videos, get_video, get_comments, save_comments,
                upsert_video, save_chat_message, get_chat_history, clear_chat_history)
from youtube_fetcher import extract_video_id, get_video_info, get_regular_comments, get_live_chat
from sentiment_analyzer import analyze, summary_stats
from llm_chat import chat as llm_chat

st.set_page_config(
    page_title="YouTube Sentiment Analyser",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CUSTOM CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans:wght@400;600&family=Noto+Sans+JP&family=Noto+Sans+KR&family=Noto+Sans+SC&family=Noto+Sans+Arabic&display=swap');
html, body, [class*="css"] {
    font-family: 'Noto Sans', 'Segoe UI', 'Arial Unicode MS', sans-serif;
}
.hero {
    background: linear-gradient(135deg, #FF0000 0%, #cc0000 50%, #880000 100%);
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 20px;
    color: white;
}
.hero h1 { margin: 0 0 4px 0; font-size: 2rem; }
.hero p  { margin: 0; opacity: 0.85; font-size: 0.95rem; }

.pill { display:inline-block; padding:3px 10px; border-radius:20px;
        font-size:0.78rem; font-weight:600; }
.pill-pos { background:#1a4731; color:#2ecc71; }
.pill-neu { background:#2d2d2d; color:#aaaaaa; }
.pill-neg { background:#4a1515; color:#e74c3c; }

.ccard { border-left:4px solid; padding:10px 14px;
         border-radius:0 8px 8px 0; margin-bottom:10px;
         background:rgba(255,255,255,0.03); }
.ccard-pos { border-color:#2ecc71; }
.ccard-neu { border-color:#666; }
.ccard-neg { border-color:#e74c3c; }
.ccard .author { font-weight:600; font-size:0.82rem; opacity:0.7; }
.ccard .body   { margin-top:4px; font-size:0.92rem; }
.ccard .footer { font-size:0.75rem; opacity:0.5; margin-top:6px; }

[data-testid="stSidebar"] .stButton button { text-align:left; font-size:0.82rem; }
</style>
""", unsafe_allow_html=True)

# CONSTANTS
COLOR_MAP = {"Positive": "#2ecc71", "Neutral": "#95a5a6", "Negative": "#e74c3c"}
PILL_CLS  = {"Positive": "pill-pos", "Neutral": "pill-neu", "Negative": "pill-neg"}

# HELPERS
def thumb_url(vid_id: str) -> str:
    return f"https://img.youtube.com/vi/{vid_id}/hqdefault.jpg"

def health_label(stats: dict) -> tuple:
    if stats["total"] == 0:
        return 0.0, "No data", "normal"
    pct = round(stats["positive"] / stats["total"] * 100, 1)
    if pct >= 55: return pct, "Positive Audience", "normal"
    if pct >= 30: return pct, "Mixed Audience", "off"
    return pct, "Negative Audience", "inverse"

def top_words(texts: list, n: int = 12) -> list:
    STOP = {"the","a","an","is","it","i","in","of","and","to","for","that","this",
            "was","on","are","with","be","at","by","he","she","we","you","they",
            "have","do","not","but","what","so","if","or","from","as","my","me",
            "your","his","her","our","just","all","will","can","more","out","up",
            "about","also","been","had","im","its","like","get","got"}
    words = re.findall(r"\w{2,}", " ".join(texts), flags=re.UNICODE)
    words = [w.lower() for w in words if not w.isdigit()]
    return Counter(w for w in words if w not in STOP and len(w) >= 2).most_common(n)

@st.cache_data(ttl=30)
def cached_comments(video_id):
    return get_comments(video_id)

@st.cache_data(ttl=30)
def cached_all_videos():
    return get_all_videos()

# SIDEBAR
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    if YT_KEY:
        st.success("✅ YouTube API Key loaded")
    else:
        st.error("❌ YOUTUBE_API_KEY missing in .env")
    if OAI_KEY:
        st.success("✅ OpenAI API Key loaded")
    else:
        st.caption("ℹ️ OPENAI_API_KEY not set — AI Chat disabled")

    if st.button("🔑 Verify YouTube Key", use_container_width=True):
        if not YT_KEY:
            st.error("No key found in .env")
        else:
            with st.spinner("Checking..."):
                try:
                    from googleapiclient.discovery import build
                    yt = build("youtube", "v3", developerKey=YT_KEY)
                    yt.videos().list(part="snippet", id="dQw4w9WgXcQ").execute()
                    st.success("✅ Key is valid!")
                except Exception as e:
                    st.error(f"❌ {e}")

    st.markdown("---")
    st.markdown("### 📂 History")
    all_videos = cached_all_videos()
    active_vid = st.session_state.get("selected_video_id")
    if all_videos:
        for v in all_videos:
            label = (v["title"] or v["video_id"])[:34]
            is_active = v["video_id"] == active_vid
            if st.button(
                f"{'▶ ' if is_active else '🎬 '}{label}",
                key=f"v_{v['video_id']}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state["selected_video_id"] = v["video_id"]
                st.cache_data.clear()
                st.rerun()
    else:
        st.caption("No videos yet.")
    st.markdown("---")
    st.caption("YouTube Sentiment Analyser · VADER + GPT")

# HERO HEADER
st.markdown("""
<div class="hero">
  <h1>🎬 YouTube Sentiment Analyser</h1>
  <p>Paste a URL &rarr; fetch comments &amp; live chat &rarr; instant sentiment analysis &rarr; AI Q&amp;A</p>
</div>
""", unsafe_allow_html=True)

# INPUT ROW
col_url, col_btn = st.columns([5, 1])
with col_url:
    url_input = st.text_input(
        "URL", placeholder="https://www.youtube.com/watch?v=...  or  https://youtu.be/...",
        label_visibility="collapsed",
    )
with col_btn:
    analyse_btn = st.button("🔍 Analyse", type="primary", use_container_width=True,
                            disabled=not YT_KEY)

with st.expander("⚙️ Fetch options", expanded=False):
    c_opt1, c_opt2 = st.columns(2)
    max_comments = c_opt1.slider("Max regular comments", 50, 500, 200, 50)
    max_chat     = c_opt2.slider("Max live chat messages", 50, 1000, 500, 50)

# ANALYSE LOGIC
if analyse_btn:
    if not url_input.strip():
        st.error("❌ Enter a YouTube URL first.")
        st.stop()

    video_id = extract_video_id(url_input.strip())
    st.session_state["selected_video_id"] = video_id
    progress = st.progress(0, text="🔍 Fetching video metadata...")

    info = get_video_info(YT_KEY, video_id)
    if not info or "error" in info:
        st.error(f"❌ Could not fetch video: {info.get('error', 'Video not found.')}")
        st.stop()

    upsert_video(video_id, url_input.strip(),
                 info.get("title",""), info.get("channel",""),
                 info.get("description",""), info.get("published",""))
    progress.progress(20, text=f"✅ Got: {info.get('title','')[:70]}")

    progress.progress(30, text="💬 Fetching comments...")
    raw_comments, comment_err = get_regular_comments(YT_KEY, video_id, max_comments)

    progress.progress(60, text="🔴 Fetching live chat...")
    raw_chat, chat_err = get_live_chat(YT_KEY, video_id, max_chat)

    progress.progress(80, text="🧠 Running sentiment analysis...")
    # Deduplicate by (normalised text, source) — keeps first occurrence
    seen, unique_raw = set(), []
    for item in raw_comments + raw_chat:
        key = (item["text"].strip().lower(), item["source"])
        if key not in seen:
            seen.add(key)
            unique_raw.append(item)
    dupes = len(raw_comments) + len(raw_chat) - len(unique_raw)
    all_raw = analyze(unique_raw)
    save_comments(video_id, all_raw)
    progress.progress(100, text="✅ Done!")
    st.cache_data.clear()

    n_c, n_l = len(raw_comments), len(raw_chat)
    if n_c == 0 and n_l == 0:
        if comment_err and ("disabled" in comment_err.lower() or "disabled" in str(comment_err).lower()):
            st.warning(f"⚠️ Comments are disabled for this video. No data to analyse.")
        elif comment_err and ("invalid" in comment_err.lower() or "key" in comment_err.lower()):
            st.error(f"❌ API key error: {comment_err}")
        else:
            st.error("❌ No data retrieved — the video may have no comments, or live chat replay is unavailable.")
    else:
        parts = ([f"**{n_c}** comments"] if n_c else []) + ([f"**{n_l}** live chat msgs"] if n_l else [])
        dupe_note = f" *(removed {dupes} duplicate{'s' if dupes != 1 else ''})*" if dupes else ""
        st.success(f"✅ Analysis complete — {' + '.join(parts)} analysed{dupe_note}.")
    if comment_err and n_c == 0 and n_l > 0: st.warning(f"⚠️ Comments: {comment_err}")
    if chat_err:    st.info(f"ℹ️ Live chat: {chat_err}")

# REQUIRE SELECTION
selected_id = st.session_state.get("selected_video_id")

if not selected_id:
    st.markdown("""
    <div style="text-align:center;padding:60px 20px;opacity:0.45;">
      <div style="font-size:4rem;">🎬</div>
      <div style="font-size:1.1rem;margin-top:12px;">
        Enter a YouTube URL above and click <strong>Analyse</strong> to get started.
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

video    = get_video(selected_id)
all_data = cached_comments(selected_id)

if not video:
    st.warning("Video not found — click Analyse."); st.stop()
if not all_data:
    st.warning("No comments saved yet — click Analyse."); st.stop()

df          = pd.DataFrame(all_data)
df_comments = df[df["source"] == "comment"].reset_index(drop=True)
df_chat     = df[df["source"] == "livechat"].reset_index(drop=True)

# VIDEO HEADER
h1, h2 = st.columns([3, 1])
with h1:
    st.markdown(f"## 📺 {video.get('title','')}")
    st.caption(
        f"Channel: **{video.get('channel','')}**  ·  "
        f"Fetched: {video.get('fetched_at','')[:16]} UTC  ·  "
        f"{len(all_data)} records"
    )
    with st.expander("📄 Description"):
        desc = video.get("description","") or "No description."
        st.write(desc[:1500] + ("..." if len(desc) > 1500 else ""))
with h2:
    st.image(thumb_url(selected_id), use_container_width=True)

st.markdown("---")

# HEALTH BANNER
ov = summary_stats(all_data)
hscore, hlabel, hdelta_type = health_label(ov)
tot = max(ov["total"], 1)

mc1, mc2, mc3, mc4, mc5 = st.columns(5)
mc1.metric("Total",          ov["total"])
mc2.metric("✅ Positive",    f"{ov['positive']} ({ov['positive']/tot*100:.0f}%)")
mc3.metric("⚪ Neutral",     f"{ov['neutral']}  ({ov['neutral']/tot*100:.0f}%)")
mc4.metric("❌ Negative",    f"{ov['negative']} ({ov['negative']/tot*100:.0f}%)")
mc5.metric("💯 Health Score", f"{hscore}%", delta=hlabel, delta_color=hdelta_type)

skipped = ov.get("hailing_skipped", 0)
if skipped:
    st.caption(f"ℹ️ {skipped} hailing / chanting comment{'s' if skipped != 1 else ''} (e.g. \"Jai …\", \"Zindabad\", \"Hurray\") excluded from sentiment scoring.")

st.markdown("---")

# TABS
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Sentiment Overview",
    "💬 Comments Browser",
    "🔍 Insights",
    "🤖 AI Chat",
])

# TAB 1 — SENTIMENT OVERVIEW
with tab1:
    c1, c2 = st.columns(2)
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
                    st.info("No live chat — not a live stream.")
            else:
                s = summary_stats(df_src.to_dict("records"))
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Total", s["total"])
                s2.metric("Pos",   s["positive"])
                s3.metric("Neu",   s["neutral"])
                s4.metric("Neg",   s["negative"])
                st.caption(f"Avg score: **{s['avg_score']:+.3f}**")
                counts = df_src["sentiment"].value_counts().reset_index()
                counts.columns = ["Sentiment", "Count"]
                fig = px.bar(counts, x="Sentiment", y="Count", color="Sentiment",
                             color_discrete_map=COLOR_MAP, text="Count",
                             title=f"{label} Distribution")
                fig.update_traces(textposition="outside")
                fig.update_layout(showlegend=False, height=280, margin=dict(t=40, b=0),
                                  plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    pc1, pc2 = st.columns(2)
    with pc1:
        st.markdown("#### Overall Distribution")
        ovc = df["sentiment"].value_counts().reset_index()
        ovc.columns = ["Sentiment", "Count"]
        fig2 = px.pie(ovc, names="Sentiment", values="Count",
                      color="Sentiment", color_discrete_map=COLOR_MAP, hole=0.45)
        fig2.update_traces(textinfo="label+percent", pull=[0.04]*3)
        fig2.update_layout(height=300, showlegend=False, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)
    with pc2:
        st.markdown("#### Score Distribution")
        fig3 = px.histogram(df, x="score", color="source", nbins=40,
                            barmode="overlay", opacity=0.75,
                            color_discrete_map={"comment": "#3498db", "livechat": "#e67e22"},
                            labels={"score": "VADER Score", "source": "Source"})
        fig3.add_vline(x=0.05,  line_dash="dash", line_color="#2ecc71",
                       annotation_text="Positive", annotation_position="top right")
        fig3.add_vline(x=-0.05, line_dash="dash", line_color="#e74c3c",
                       annotation_text="Negative")
        fig3.update_layout(height=300, margin=dict(t=10),
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig3, use_container_width=True)

    # Sentiment trend
    if len(df) >= 5:
        st.markdown("---")
        st.markdown("#### Sentiment Trend (comment order)")
        df_t = df.reset_index(drop=True).reset_index().rename(columns={"index": "order"})
        df_t["rolling"] = df_t["score"].rolling(10, min_periods=1).mean()
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=df_t["order"], y=df_t["score"], mode="markers",
            marker=dict(size=5, opacity=0.35, color=df_t["score"],
                        colorscale=[[0,"#e74c3c"],[0.5,"#95a5a6"],[1,"#2ecc71"]],
                        cmin=-1, cmax=1), name="Score",
        ))
        fig4.add_trace(go.Scatter(
            x=df_t["order"], y=df_t["rolling"], mode="lines",
            line=dict(color="#f39c12", width=3), name="10-comment avg",
        ))
        fig4.add_hline(y=0, line_dash="dot", line_color="white", opacity=0.25)
        fig4.update_layout(height=260, margin=dict(t=10, b=0),
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           xaxis_title="Comment index", yaxis_title="VADER score")
        st.plotly_chart(fig4, use_container_width=True)

    # Top comment cards
    st.markdown("---")
    tp1, tp2 = st.columns(2)
    for col, subset, heading, css in [
        (tp1, df.nlargest(5, "score"),  "🟢 Top 5 Most Positive", "ccard-pos"),
        (tp2, df.nsmallest(5, "score"), "🔴 Top 5 Most Negative", "ccard-neg"),
    ]:
        with col:
            st.markdown(f"#### {heading}")
            for _, r in subset.iterrows():
                pc = PILL_CLS.get(r["sentiment"], "pill-neu")
                safe_author = h(str(r["author"])[:40])
                safe_text   = h(str(r["text"])[:200])
                safe_src    = h(str(r["source"]))
                safe_sent   = h(str(r["sentiment"]))
                st.markdown(f"""
<div class="ccard {css}">
  <div class="author">
    <span class="pill {pc}">{safe_sent}</span>
    &nbsp;<strong>{safe_author}</strong>
    &nbsp;&middot;&nbsp;<code>{r['score']:+.2f}</code>
    &nbsp;&middot;&nbsp;{safe_src}
  </div>
  <div class="body">{safe_text}</div>
  <div class="footer">👍 {int(r.get('likes', 0))} likes</div>
</div>""", unsafe_allow_html=True)

# TAB 2 — COMMENTS BROWSER
with tab2:
    f1, f2, f3, f4 = st.columns([2, 2, 2, 3])
    src_filter  = f1.selectbox("Source",    ["All", "comment", "livechat"])
    sent_filter = f2.selectbox("Sentiment", ["All", "Positive", "Neutral", "Negative"])
    sort_map    = {"Score (high to low)": ("score", False), "Score (low to high)": ("score", True),
                   "Likes (high to low)": ("likes", False), "Default": (None, False)}
    sort_choice = f3.selectbox("Sort by", list(sort_map.keys()))
    search_term = f4.text_input("🔎 Search comments")

    filtered = df.copy()
    if src_filter  != "All": filtered = filtered[filtered["source"]    == src_filter]
    if sent_filter != "All": filtered = filtered[filtered["sentiment"] == sent_filter]
    if search_term:
        filtered = filtered[filtered["text"].str.contains(search_term, case=False, na=False)]
    sc, sa = sort_map[sort_choice]
    if sc: filtered = filtered.sort_values(sc, ascending=sa)

    st.caption(f"Showing **{len(filtered)}** of **{len(df)}** records")
    st.dataframe(
        filtered[["source","sentiment","score","author","text","likes"]],
        use_container_width=True, height=480,
        column_config={
            "score":     st.column_config.NumberColumn("Score",   format="%.3f"),
            "sentiment": st.column_config.TextColumn("Sentiment"),
            "source":    st.column_config.TextColumn("Source"),
            "text":      st.column_config.TextColumn("Comment",  width="large"),
            "likes":     st.column_config.NumberColumn("Likes"),
            "author":    st.column_config.TextColumn("Author"),
        }
    )
    dl1, dl2, _ = st.columns([1, 1, 4])
    csv = filtered.to_csv(index=False).encode("utf-8")
    dl1.download_button("Download CSV", csv, f"{selected_id}_comments.csv", "text/csv",
                        use_container_width=True)
    md_rows = "\n".join(
        f"| {r.sentiment} | {r.score:+.2f} | {str(r.author)[:20]} | {str(r.text)[:100]} |"
        for r in filtered.itertuples()
    )
    md_full = f"| Sentiment | Score | Author | Comment |\n|---|---|---|---|\n{md_rows}"
    dl2.download_button("Download Markdown", md_full.encode("utf-8"), f"{selected_id}_comments.md",
                        "text/markdown", use_container_width=True)

# TAB 3 — INSIGHTS
with tab3:
    i1, i2 = st.columns(2)

    with i1:
        st.markdown("#### 👥 Top Commenters")
        if df_comments.empty:
            st.info("No regular comments.")
        else:
            auth = (df_comments.groupby("author")
                    .agg(count=("text","count"), avg_score=("score","mean"),
                         total_likes=("likes","sum"))
                    .sort_values("count", ascending=False).head(15).reset_index())
            auth["avg_score"] = auth["avg_score"].round(3)
            fig_a = px.bar(auth, x="count", y="author", orientation="h",
                           color="avg_score",
                           color_continuous_scale=["#e74c3c","#95a5a6","#2ecc71"],
                           color_continuous_midpoint=0, range_color=[-0.5, 0.5],
                           labels={"count":"# Comments","author":"","avg_score":"Avg Score"},
                           title="Top 15 commenters (colour = avg sentiment)")
            fig_a.update_layout(height=380, margin=dict(t=40,b=0),
                                yaxis=dict(autorange="reversed"),
                                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_a, use_container_width=True)

    with i2:
        st.markdown("#### 🔤 Top Words by Sentiment")
        wt1, wt2 = st.tabs(["Positive", "Negative"])
        for wtab, slabel in [(wt1, "Positive"), (wt2, "Negative")]:
            with wtab:
                subset_texts = df[df["sentiment"] == slabel]["text"].tolist()
                if not subset_texts:
                    st.info(f"No {slabel.lower()} comments.")
                else:
                    words = top_words(subset_texts)
                    if words:
                        wdf = pd.DataFrame(words, columns=["Word","Count"])
                        fig_w = px.bar(wdf, x="Count", y="Word", orientation="h",
                                       color_discrete_sequence=[COLOR_MAP[slabel]])
                        fig_w.update_layout(yaxis=dict(autorange="reversed"),
                                            height=340, margin=dict(t=10,b=0),
                                            plot_bgcolor="rgba(0,0,0,0)",
                                            paper_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig_w, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Comments vs Live Chat Comparison")
    if not df_comments.empty and not df_chat.empty:
        comp = []
        for dfs, lbl in [(df_comments,"Comments"),(df_chat,"Live Chat")]:
            s = summary_stats(dfs.to_dict("records"))
            t = max(s["total"],1)
            comp.append({"Source":lbl,
                         "Positive":round(s["positive"]/t*100,1),
                         "Neutral": round(s["neutral"]/t*100,1),
                         "Negative":round(s["negative"]/t*100,1)})
        comp_df = pd.DataFrame(comp).melt("Source", var_name="Sentiment", value_name="Pct")
        fig_c = px.bar(comp_df, x="Source", y="Pct", color="Sentiment",
                       barmode="group", color_discrete_map=COLOR_MAP,
                       labels={"Pct":"% of messages"},
                       title="Side-by-side breakdown")
        fig_c.update_layout(height=320, plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_c, use_container_width=True)
    else:
        st.info("Need both comment types to compare — only one source available for this video.")

# TAB 4 — AI CHAT
with tab4:
    st.subheader("🤖 Ask AI about this Video's Audience")

    if not OAI_KEY:
        st.info(
            "**OpenAI API Key not set.**  \n"
            "Add `OPENAI_API_KEY=your_key` to your `.env` file and restart the app.  \n"
            "Get a key at [platform.openai.com](https://platform.openai.com)."
        )
        with st.expander("💡 Things you could ask (once key is set)"):
            st.markdown("""
- *What are viewers most unhappy about?*
- *Summarise the top 5 themes in the comments.*
- *Are live chat viewers more positive than commenters?*
- *Which comment got the most likes and why?*
- *What would you recommend to the creator?*
""")
    else:
        st.markdown("**Quick questions:**")
        qcols = st.columns(3)
        suggestions = [
            "Summarise viewer sentiment in 3 bullet points.",
            "What are viewers most unhappy about?",
            "What topics come up most in positive comments?",
            "Are live chat viewers more positive than commenters?",
            "List the top concerns raised by the audience.",
            "What would you recommend to the creator?",
        ]
        for i, q in enumerate(suggestions):
            if qcols[i % 3].button(q, key=f"q_{i}", use_container_width=True):
                st.session_state["prefill_chat"] = q

        st.markdown("---")

        prefill  = st.session_state.pop("prefill_chat", None)
        user_msg = st.chat_input("Ask anything about this video's comments or sentiment...") or prefill

        if user_msg:
            with st.chat_message("user"):
                st.write(user_msg)
            save_chat_message(selected_id, "user", user_msg)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        reply = llm_chat(
                            api_key=OAI_KEY,
                            video_info=video,
                            comments=all_data,
                            history=get_chat_history(selected_id),
                            user_message=user_msg,
                        )
                        st.write(reply)
                        save_chat_message(selected_id, "assistant", reply)
                    except Exception as e:
                        st.error(f"LLM error: {e}")
            st.rerun()

        history = get_chat_history(selected_id)
        if history:
            ccol, _ = st.columns([1, 5])
            if ccol.button("🗑️ Clear History", use_container_width=True):
                clear_chat_history(selected_id)
                st.rerun()

        for msg in reversed(history):
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

