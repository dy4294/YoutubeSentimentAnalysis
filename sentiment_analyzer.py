
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
_analyzer = SentimentIntensityAnalyzer()

# Hailing / chanting patterns — single-phrase cheers with no substantive content.
# These are treated as Neutral and excluded from sentiment scoring.
_HAIL_PATTERN = re.compile(
    r"^[\W]*(jai|जय|జై|ஜெய்|zindabad|زندہ باد|viva|урааа|bravo|hurray|hooray|hail|long live|\U0001F64C|\U0001F44F)[\w\s!.,'\U00010000-\U0010FFFF]{0,40}$",
    re.IGNORECASE | re.UNICODE,
)

def is_hailing(text: str) -> bool:
    """Return True for short chanting / hailing comments that carry no analytical value."""
    stripped = text.strip()
    # Must be short (≤12 words) to qualify
    if len(stripped.split()) > 12:
        return False
    return bool(_HAIL_PATTERN.match(stripped))

def analyze(items: list) -> list:
    results = []
    for item in items:
        if is_hailing(item["text"]):
            results.append({**item, "sentiment": "Neutral", "score": 0.0, "hailing": True})
            continue
        compound = _analyzer.polarity_scores(item["text"])["compound"]
        label = "Positive" if compound >= 0.05 else "Negative" if compound <= -0.05 else "Neutral"
        results.append({**item, "sentiment": label, "score": compound, "hailing": False})
    return results

def summary_stats(comments: list) -> dict:
    if not comments: return {}
    from collections import Counter
    # Exclude hailing comments from sentiment statistics
    scored = [c for c in comments if not c.get("hailing", False)]
    if not scored:
        return {"total": 0, "positive": 0, "neutral": 0, "negative": 0, "avg_score": 0.0,
                "hailing_skipped": len(comments)}
    counts = Counter(c["sentiment"] for c in scored)
    avg = sum(c["score"] for c in scored) / len(scored)
    return {"total": len(scored), "positive": counts.get("Positive", 0),
            "neutral": counts.get("Neutral", 0), "negative": counts.get("Negative", 0),
            "avg_score": round(avg, 3), "hailing_skipped": len(comments) - len(scored)}
