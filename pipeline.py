import hashlib
import re
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd
import torch
import trafilatura
from dateutil import parser as date_parser
from ddgs import DDGS
from transformers import AutoModelForSequenceClassification, AutoTokenizer

EXCLUDED_DOMAINS = {
    "linkedin.com", "facebook.com", "instagram.com", "x.com", "twitter.com",
    "youtube.com", "reddit.com", "wikipedia.org", "wikidata.org", "glassdoor.com"
}
TRACKING_KEYS = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "gclid", "fbclid"}

def normalize_url(url):
    try:
        p = urlsplit(url)
        host = (p.hostname or "").lower().removeprefix("www.")
        query = urlencode([(k, v) for k, v in parse_qsl(p.query) if k.lower() not in TRACKING_KEYS])
        path = re.sub(r"/+$", "", p.path) or "/"
        return urlunsplit((p.scheme.lower() or "https", host, path, query, ""))
    except Exception:
        return url

def excluded(url):
    host = (urlsplit(url).hostname or "").lower().removeprefix("www.")
    return any(host == d or host.endswith("." + d) for d in EXCLUDED_DOMAINS)

def parse_date(value):
    if not value:
        return None
    try:
        dt = date_parser.parse(str(value))
        if not dt.tzinfo:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None

def make_cache_key(company, industry, region, days):
    raw = f"{company.strip().casefold()}|{industry.strip().casefold()}|{region}|{days}"
    return hashlib.sha256(raw.encode()).hexdigest()

def search_news(company, industry="", region="wt-wt", days=30, max_results=60):
    suffix = f" {industry.strip()}" if industry.strip() else ""
    queries = [f'"{company}"{suffix}', f'"{company}" company news{suffix}']
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    unique = {}
    for query in queries:
        try:
            rows = DDGS(timeout=15).news(
                query=query, region=region, safesearch="moderate",
                timelimit="m", max_results=max(10, max_results // len(queries))
            )
        except Exception:
            rows = []
        for r in rows or []:
            url = r.get("url") or r.get("href") or ""
            if not url or excluded(url):
                continue
            published = parse_date(r.get("date"))
            if published and published < cutoff:
                continue
            normalized = normalize_url(url)
            unique.setdefault(normalized, {
                "url": url,
                "normalized_url": normalized,
                "title": (r.get("title") or "Untitled").strip(),
                "source": (r.get("source") or urlsplit(url).hostname or "Unknown").strip(),
                "published_at": published.isoformat() if published else None,
                "snippet": (r.get("body") or r.get("snippet") or "").strip(),
            })
    return list(unique.values())[:max_results]

def extract_article(url, title, snippet):
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded, include_comments=False, include_tables=False) or ""
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) >= 400:
                return text, "trafilatura"
    except Exception:
        pass
    return f"{title}. {snippet}".strip(), "ddgs_snippet"

def relevance_score(company, title, snippet, text):
    company_cf = company.casefold().strip()
    exact = re.compile(rf"(?<!\w){re.escape(company_cf)}(?!\w)", re.I)
    score = 0
    if exact.search(title or ""):
        score += 4
    if exact.search(snippet or ""):
        score += 2
    matches = exact.findall((text or "")[:12000])
    score += min(3, len(matches))
    return score

def sentiment_input(company, title, snippet, text, max_chars=7000):
    sentences = re.split(r"(?<=[.!?])\s+", text or "")
    picked = []
    name = company.casefold()
    for i, sentence in enumerate(sentences):
        if name in sentence.casefold():
            picked.extend(sentences[max(0, i-1):min(len(sentences), i+2)])
    context = " ".join(dict.fromkeys(picked))
    seed = f"{title}. {snippet}. {context or (text or '')}"
    return re.sub(r"\s+", " ", seed).strip()[:max_chars]

class FinBert:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(self.device)
        self.model.eval()
        self.id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}

    def predict(self, texts, batch_size=8):
        outputs = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            encoded = self.tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.inference_mode():
                probs = torch.softmax(self.model(**encoded).logits, dim=-1).cpu().numpy()
            for row in probs:
                scores = {self.id2label[i]: float(row[i]) for i in range(len(row))}
                label = max(scores, key=scores.get)
                outputs.append({
                    "sentiment_label": label,
                    "positive": scores.get("positive", 0.0),
                    "neutral": scores.get("neutral", 0.0),
                    "negative": scores.get("negative", 0.0),
                    "signed_score": scores.get("positive", 0.0) - scores.get("negative", 0.0),
                })
        return outputs

def analyze(company, industry="", region="wt-wt", days=30, max_results=60, progress=None):
    articles = search_news(company, industry, region, days, max_results)
    retained = []
    for idx, article in enumerate(articles):
        if progress:
            progress(0.1 + 0.45 * ((idx + 1) / max(len(articles), 1)), f"Extracting article {idx + 1}/{len(articles)}")
        text, method = extract_article(article["url"], article["title"], article["snippet"])
        score = relevance_score(company, article["title"], article["snippet"], text)
        if score < 2:
            continue
        article.update(article_text=text, extraction_method=method, relevance_score=score)
        retained.append(article)
    if not retained:
        return []
    if progress:
        progress(0.62, "Loading FinBERT and analyzing sentiment")
    model = FinBert()
    inputs = [sentiment_input(company, a["title"], a["snippet"], a["article_text"]) for a in retained]
    predictions = model.predict(inputs)
    for article, prediction in zip(retained, predictions):
        article.update(prediction)
    if progress:
        progress(1.0, "Analysis complete")
    return retained

def to_frame(articles):
    df = pd.DataFrame(articles)
    if not df.empty:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
        df["date"] = df["published_at"].dt.date
    return df
