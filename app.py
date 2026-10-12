from datetime import datetime, timezone
from html import escape

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from database import get_articles, get_cached_run, init_db, replace_run
from pipeline import analyze, make_cache_key, to_frame


# -----------------------------------------------------------------------------
# Page configuration and database
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Company Intelligence",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)
init_db()


# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------
DEFAULT_STATE = {
    "articles": [],
    "company": "",
    "industry": "",
    "region": "wt-wt",
    "days": 30,
    "max_results": 50,
    "cache_hours": 24,
    "cache_status": "",
    "searched": False,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Hide sidebar and reduce native app chrome */
    [data-testid="stSidebar"] {display: none;}
    [data-testid="collapsedControl"] {display: none;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .stApp {
        background:
            radial-gradient(circle at 50% -10%, rgba(66, 133, 244, 0.10), transparent 30%),
            var(--background-color);
    }

    .block-container {
        max-width: 1240px;
        padding-top: 1.2rem;
        padding-bottom: 4rem;
    }

    /* Homepage */
    .home-shell {
        min-height: 55vh;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        padding-top: 6vh;
    }

    .brand-mark {
        width: 72px;
        height: 72px;
        border-radius: 22px;
        display: grid;
        place-items: center;
        margin: 0 auto 1rem auto;
        font-size: 2.1rem;
        background: linear-gradient(135deg, #4285f4 0%, #8b5cf6 55%, #ec4899 100%);
        box-shadow: 0 16px 38px rgba(66, 133, 244, 0.22);
    }

    .home-title {
        margin: 0;
        font-size: clamp(2.4rem, 5.2vw, 4.7rem);
        font-weight: 780;
        line-height: 1.05;
        letter-spacing: -0.045em;
    }

    .home-subtitle {
        max-width: 720px;
        margin: 1rem auto 2rem auto;
        color: #8d96a7;
        font-size: 1.08rem;
        line-height: 1.65;
    }

    .search-caption {
        text-align: center;
        color: #8992a3;
        font-size: 0.88rem;
        margin-top: 0.6rem;
    }

    /* Search inputs */
    div[data-testid="stTextInput"] input {
        min-height: 58px;
        border-radius: 999px;
        padding-left: 24px;
        padding-right: 24px;
        font-size: 1.05rem;
        border: 1px solid rgba(137, 146, 163, 0.42);
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.10);
    }

    div[data-testid="stTextInput"] input:focus {
        border-color: #4285f4;
        box-shadow: 0 0 0 3px rgba(66, 133, 244, 0.14), 0 10px 30px rgba(15, 23, 42, 0.10);
    }

    div.stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        min-height: 48px;
        border-radius: 999px;
        font-weight: 700;
        border: 0;
    }

    /* Results header */
    .results-brand {
        font-size: 1.35rem;
        font-weight: 780;
        letter-spacing: -0.025em;
        margin-top: 0.55rem;
        white-space: nowrap;
    }

    .result-heading {
        margin: 1.5rem 0 0.2rem 0;
        font-size: clamp(2rem, 4vw, 3.25rem);
        font-weight: 780;
        letter-spacing: -0.04em;
    }

    .muted {
        color: #8d96a7;
    }

    .status-pill {
        display: inline-block;
        padding: 0.38rem 0.72rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        background: rgba(66, 133, 244, 0.12);
        color: #4285f4;
        margin: 0.45rem 0 0.5rem 0;
    }

    /* KPI cards */
    div[data-testid="stMetric"] {
        background: rgba(127, 127, 127, 0.055);
        border: 1px solid rgba(137, 146, 163, 0.22);
        padding: 1rem 1.05rem;
        border-radius: 18px;
        min-height: 118px;
    }

    div[data-testid="stMetricLabel"] {
        color: #8d96a7;
    }

    /* Article search-result cards */
    .result-card {
        padding: 1.15rem 0.15rem 1.25rem 0.15rem;
        border-bottom: 1px solid rgba(137, 146, 163, 0.18);
    }

    .result-source {
        color: #8d96a7;
        font-size: 0.84rem;
        margin-bottom: 0.28rem;
    }

    .result-title a {
        color: #4b8df8;
        text-decoration: none;
        font-size: 1.18rem;
        line-height: 1.35;
        font-weight: 690;
    }

    .result-title a:hover {
        text-decoration: underline;
    }

    .result-snippet {
        margin-top: 0.48rem;
        color: #abb3c1;
        line-height: 1.55;
        font-size: 0.93rem;
    }

    .badge {
        display: inline-block;
        border-radius: 999px;
        padding: 0.20rem 0.54rem;
        margin-left: 0.38rem;
        font-size: 0.72rem;
        font-weight: 800;
        text-transform: capitalize;
    }

    .positive {background: rgba(34, 197, 94, 0.14); color: #22c55e;}
    .neutral {background: rgba(148, 163, 184, 0.15); color: #94a3b8;}
    .negative {background: rgba(239, 68, 68, 0.14); color: #ef4444;}

    .section-note {
        color: #8d96a7;
        font-size: 0.91rem;
    }

    /* Source cards */
    .source-card {
        border: 1px solid rgba(137, 146, 163, 0.22);
        background: rgba(127, 127, 127, 0.045);
        border-radius: 16px;
        padding: 1rem;
        margin-bottom: 0.75rem;
    }

    .source-name {font-weight: 750; font-size: 1.02rem;}
    .source-count {color: #8d96a7; font-size: 0.88rem; margin-top: 0.25rem;}

    @media (max-width: 760px) {
        .home-shell {padding-top: 2vh; min-height: 45vh;}
        .block-container {padding-left: 1rem; padding-right: 1rem;}
        .home-title {font-size: 2.55rem;}
        .results-brand {text-align: center;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def execute_search(
    company: str,
    industry: str,
    region: str,
    days: int,
    max_results: int,
    cache_hours: int,
    force_refresh: bool,
) -> None:
    """Load a cached run or perform a new DDGS, extraction, and FinBERT run."""
    company = company.strip()
    industry = industry.strip()

    if not company:
        st.warning("Enter a company name before searching.")
        return

    cache_key = make_cache_key(company, industry, region, days)
    cached = None if force_refresh else get_cached_run(cache_key, cache_hours)

    if cached:
        articles = get_articles(cached["id"])
        cache_status = f"Loaded from SQLite cache created {cached['created_at']}"
    else:
        progress_bar = st.progress(0, text="Searching recent news...")

        def update_progress(value, text):
            progress_bar.progress(min(float(value), 1.0), text=text)

        try:
            articles = analyze(
                company=company,
                industry=industry,
                region=region,
                days=days,
                max_results=max_results,
                progress=update_progress,
            )
        except Exception as exc:
            progress_bar.empty()
            st.error(f"Analysis failed: {exc}")
            return

        progress_bar.empty()

        run_id = replace_run(
            cache_key=cache_key,
            company=company,
            industry=industry,
            region=region,
            days=days,
            created_at=datetime.now(timezone.utc).isoformat(),
            articles=articles,
        )
        articles = get_articles(run_id)
        cache_status = "Fresh internet search saved to SQLite"

    st.session_state.articles = articles
    st.session_state.company = company
    st.session_state.industry = industry
    st.session_state.region = region
    st.session_state.days = days
    st.session_state.max_results = max_results
    st.session_state.cache_hours = cache_hours
    st.session_state.cache_status = cache_status
    st.session_state.searched = True


def clear_results() -> None:
    st.session_state.articles = []
    st.session_state.company = ""
    st.session_state.cache_status = ""
    st.session_state.searched = False


def sentiment_badge(label: str) -> str:
    safe_label = str(label or "neutral").lower()
    css_class = safe_label if safe_label in {"positive", "neutral", "negative"} else "neutral"
    return f'<span class="badge {css_class}">{escape(safe_label)}</span>'


def safe_text(value, fallback="") -> str:
    if value is None or pd.isna(value):
        return fallback
    return str(value)


def render_article_card(row) -> None:
    title = escape(safe_text(row.get("title"), "Untitled article"))
    url = escape(safe_text(row.get("url"), "#"), quote=True)
    source = escape(safe_text(row.get("source"), "Unknown source"))
    snippet = escape(safe_text(row.get("snippet"), "No snippet available."))
    label = safe_text(row.get("sentiment_label"), "neutral").lower()

    published = row.get("published_at")
    if pd.isna(published):
        date_text = "Date unavailable"
    else:
        date_text = pd.Timestamp(published).strftime("%d %b %Y")

    confidence_values = [row.get("positive", 0), row.get("neutral", 0), row.get("negative", 0)]
    confidence_values = [float(v or 0) for v in confidence_values]
    confidence = max(confidence_values) if confidence_values else 0

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-source">
                {source} · {escape(date_text)}
                {sentiment_badge(label)}
                <span class="muted"> · {confidence:.0%} confidence</span>
            </div>
            <div class="result-title">
                <a href="{url}" target="_blank" rel="noopener noreferrer">{title}</a>
            </div>
            <div class="result-snippet">{snippet}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_reputation_gauge(score: int):
    color = "#22c55e" if score >= 60 else "#ef4444" if score < 40 else "#f59e0b"
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "/100", "font": {"size": 38}},
            title={"text": "Media reputation index"},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": color},
                "bgcolor": "rgba(127,127,127,0.08)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "rgba(239,68,68,0.10)"},
                    {"range": [40, 60], "color": "rgba(245,158,11,0.10)"},
                    {"range": [60, 100], "color": "rgba(34,197,94,0.10)"},
                ],
            },
        )
    )
    figure.update_layout(height=310, margin=dict(l=25, r=25, t=65, b=25))
    return figure


# -----------------------------------------------------------------------------
# Google-style homepage
# -----------------------------------------------------------------------------
if not st.session_state.searched:
    st.markdown(
        """
        <div class="home-shell">
            <div class="brand-mark">⌕</div>
            <h1 class="home-title">Company Intelligence</h1>
            <div class="home-subtitle">
                Search any company to explore recent media sentiment, reputation signals,
                news coverage, and the sources shaping the narrative.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    center_left, center, center_right = st.columns([1.6, 5.8, 1.6])

    with center:
        with st.form("home_search_form", clear_on_submit=False):
            company_query = st.text_input(
                "Company search",
                placeholder="Search any company...",
                label_visibility="collapsed",
            )

            with st.expander("Advanced search", expanded=False):
                adv1, adv2 = st.columns(2)
                industry = adv1.text_input(
                    "Industry or disambiguation",
                    placeholder="Example: private equity",
                )
                region = adv2.selectbox(
                    "Search region",
                    ["wt-wt", "us-en", "uk-en", "in-en"],
                    format_func={
                        "wt-wt": "Worldwide",
                        "us-en": "United States",
                        "uk-en": "United Kingdom",
                        "in-en": "India",
                    }.get,
                )

                adv3, adv4 = st.columns(2)
                days = adv3.slider("Recent period", 7, 30, 30)
                max_results = adv4.slider("Maximum results", 10, 100, 50, step=10)

                adv5, adv6 = st.columns(2)
                cache_hours = adv5.slider("Cache duration in hours", 1, 168, 24)
                force_refresh = adv6.checkbox("Refresh from internet")

            button_left, button_center, button_right = st.columns([2.2, 2, 2.2])
            with button_center:
                submitted = st.form_submit_button(
                    "Search company",
                    type="primary",
                    use_container_width=True,
                )

        st.markdown(
            '<div class="search-caption">Recent articles · Firm-targeted FinBERT sentiment · Transparent source links</div>',
            unsafe_allow_html=True,
        )

        if submitted:
            execute_search(
                company=company_query,
                industry=industry,
                region=region,
                days=days,
                max_results=max_results,
                cache_hours=cache_hours,
                force_refresh=force_refresh,
            )
            if st.session_state.searched:
                st.rerun()

    st.stop()


# -----------------------------------------------------------------------------
# Search results header
# -----------------------------------------------------------------------------
header_brand, header_search, header_action = st.columns([1.3, 5.8, 1.1], vertical_alignment="center")

with header_brand:
    st.markdown('<div class="results-brand">⌕ Company Intelligence</div>', unsafe_allow_html=True)

with header_search:
    with st.form("top_search_form", clear_on_submit=False):
        search_col, submit_col = st.columns([6.8, 1.2], vertical_alignment="center")
        with search_col:
            top_company = st.text_input(
                "Search another company",
                value=st.session_state.company,
                label_visibility="collapsed",
                placeholder="Search another company...",
            )
        with submit_col:
            top_submitted = st.form_submit_button("Search", type="primary", use_container_width=True)

with header_action:
    if st.button("New search", use_container_width=True):
        clear_results()
        st.rerun()

with st.expander("Search settings", expanded=False):
    settings1, settings2, settings3 = st.columns(3)
    result_industry = settings1.text_input(
        "Industry or disambiguation",
        value=st.session_state.industry,
    )
    result_region = settings2.selectbox(
        "Region",
        ["wt-wt", "us-en", "uk-en", "in-en"],
        index=["wt-wt", "us-en", "uk-en", "in-en"].index(st.session_state.region),
    )
    result_days = settings3.slider("Recent period", 7, 30, st.session_state.days)

    settings4, settings5, settings6 = st.columns(3)
    result_max = settings4.slider("Maximum results", 10, 100, st.session_state.max_results, step=10)
    result_cache = settings5.slider("Cache hours", 1, 168, st.session_state.cache_hours)
    result_refresh = settings6.checkbox("Refresh from internet", key="result_refresh")

if top_submitted:
    execute_search(
        company=top_company,
        industry=result_industry,
        region=result_region,
        days=result_days,
        max_results=result_max,
        cache_hours=result_cache,
        force_refresh=result_refresh,
    )
    st.rerun()


# -----------------------------------------------------------------------------
# Prepare result data
# -----------------------------------------------------------------------------
articles = st.session_state.articles
company_name = st.session_state.company

if not articles:
    st.markdown(f'<h1 class="result-heading">{escape(company_name)}</h1>', unsafe_allow_html=True)
    st.warning(
        "No sufficiently relevant recent articles were retained. "
        "Try adding an industry qualifier in Search settings or refresh the search."
    )
    st.stop()

df = to_frame(articles)

if df.empty:
    st.warning("The search completed, but no analyzable article records were available.")
    st.stop()

for column in ["positive", "neutral", "negative", "signed_score", "relevance_score"]:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

positive_count = int((df["sentiment_label"] == "positive").sum())
neutral_count = int((df["sentiment_label"] == "neutral").sum())
negative_count = int((df["sentiment_label"] == "negative").sum())
mean_score = float(df["signed_score"].mean())
reputation_index = round(max(0, min(100, 50 + (50 * mean_score))))
unique_sources = int(df["source"].nunique())

st.markdown(f'<h1 class="result-heading">{escape(company_name)}</h1>', unsafe_allow_html=True)
st.markdown(
    f'<div class="status-pill">{escape(st.session_state.cache_status)}</div>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="section-note">Analysis of {len(df)} relevant article(s) from the selected recent period. '
    'Scores describe observed media sentiment, not an objective measure of the company.</div>',
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
metric1, metric2, metric3, metric4, metric5 = st.columns(5)
metric1.metric("Media reputation", f"{reputation_index}/100", f"{mean_score:+.2f} net sentiment")
metric2.metric("Articles analyzed", f"{len(df):,}")
metric3.metric("Positive", f"{positive_count:,}")
metric4.metric("Negative", f"{negative_count:,}")
metric5.metric("Unique sources", f"{unique_sources:,}")


# -----------------------------------------------------------------------------
# Results tabs
# -----------------------------------------------------------------------------
overview_tab, news_tab, sources_tab, sentiment_tab, methodology_tab = st.tabs(
    ["Overview", "News", "Sources", "Sentiment", "Methodology"]
)

with overview_tab:
    overview_left, overview_right = st.columns([1, 1.35])

    with overview_left:
        st.plotly_chart(
            build_reputation_gauge(reputation_index),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with overview_right:
        distribution = (
            df["sentiment_label"]
            .value_counts()
            .reindex(["positive", "neutral", "negative"], fill_value=0)
            .reset_index()
        )
        distribution.columns = ["Sentiment", "Articles"]

        donut = px.pie(
            distribution,
            names="Sentiment",
            values="Articles",
            hole=0.58,
            color="Sentiment",
            color_discrete_map={
                "positive": "#22c55e",
                "neutral": "#94a3b8",
                "negative": "#ef4444",
            },
            title="Sentiment distribution",
        )
        donut.update_traces(textposition="inside", textinfo="percent+label")
        donut.update_layout(height=310, margin=dict(l=20, r=20, t=60, b=20), legend_title_text="")
        st.plotly_chart(donut, use_container_width=True, config={"displayModeBar": False})

    st.subheader("Latest coverage")
    latest_rows = df.sort_values("published_at", ascending=False, na_position="last").head(5)
    for _, article_row in latest_rows.iterrows():
        render_article_card(article_row)

with news_tab:
    filter1, filter2, filter3 = st.columns([1.2, 1.7, 1.2])
    selected_sentiments = filter1.multiselect(
        "Sentiment",
        ["positive", "neutral", "negative"],
        default=["positive", "neutral", "negative"],
    )

    source_options = sorted(df["source"].dropna().astype(str).unique().tolist())
    selected_sources = filter2.multiselect("Sources", source_options)

    keyword = filter3.text_input("Filter headlines", placeholder="Keyword...")

    shown = df[df["sentiment_label"].isin(selected_sentiments)].copy()
    if selected_sources:
        shown = shown[shown["source"].isin(selected_sources)]
    if keyword.strip():
        shown = shown[
            shown["title"].fillna("").str.contains(keyword.strip(), case=False, regex=False)
            | shown["snippet"].fillna("").str.contains(keyword.strip(), case=False, regex=False)
        ]

    shown = shown.sort_values("published_at", ascending=False, na_position="last")
    st.caption(f"Showing {len(shown)} article(s)")

    if shown.empty:
        st.info("No articles match the selected filters.")
    else:
        for _, article_row in shown.iterrows():
            render_article_card(article_row)

        download_df = shown.copy()
        download_df["published_at"] = download_df["published_at"].astype(str)
        st.download_button(
            "Download filtered results as CSV",
            data=download_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{company_name.replace(' ', '_')}_sentiment.csv",
            mime="text/csv",
        )

with sources_tab:
    source_summary = (
        df.groupby("source", dropna=False)
        .agg(
            articles=("url", "count"),
            average_sentiment=("signed_score", "mean"),
            positive=("sentiment_label", lambda values: int((values == "positive").sum())),
            negative=("sentiment_label", lambda values: int((values == "negative").sum())),
        )
        .reset_index()
        .sort_values(["articles", "average_sentiment"], ascending=[False, False])
    )

    source_chart = px.bar(
        source_summary.head(15),
        x="articles",
        y="source",
        orientation="h",
        color="average_sentiment",
        color_continuous_scale=["#ef4444", "#94a3b8", "#22c55e"],
        range_color=[-1, 1],
        title="Most visible sources",
        labels={"articles": "Articles", "source": "Source", "average_sentiment": "Avg. sentiment"},
    )
    source_chart.update_layout(height=max(360, 34 * min(len(source_summary), 15)), yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(source_chart, use_container_width=True)

    cards = st.columns(3)
    for index, row in source_summary.head(12).reset_index(drop=True).iterrows():
        value = float(row["average_sentiment"])
        sentiment_word = "Positive" if value > 0.15 else "Negative" if value < -0.15 else "Mixed"
        with cards[index % 3]:
            st.markdown(
                f"""
                <div class="source-card">
                    <div class="source-name">{escape(safe_text(row['source'], 'Unknown source'))}</div>
                    <div class="source-count">
                        {int(row['articles'])} article(s) · {sentiment_word}<br>
                        {int(row['positive'])} positive · {int(row['negative'])} negative
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

with sentiment_tab:
    trend_df = df.dropna(subset=["date"]).copy()
    trend_df = trend_df.groupby("date", as_index=False).agg(
        average_sentiment=("signed_score", "mean"),
        articles=("url", "count"),
    )

    if trend_df.empty:
        st.info("Publication dates were unavailable, so a sentiment trend could not be created.")
    else:
        trend_chart = px.line(
            trend_df,
            x="date",
            y="average_sentiment",
            markers=True,
            hover_data={"articles": True, "date": True, "average_sentiment": ":.3f"},
            title="Average media sentiment over time",
            labels={"date": "Date", "average_sentiment": "Positive minus negative probability"},
        )
        trend_chart.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
        trend_chart.update_yaxes(range=[-1, 1])
        trend_chart.update_layout(height=430)
        st.plotly_chart(trend_chart, use_container_width=True)

    st.subheader("Article confidence")
    confidence_df = df.copy()
    confidence_df["confidence"] = confidence_df[["positive", "neutral", "negative"]].max(axis=1)
    confidence_df["published"] = confidence_df["published_at"].dt.strftime("%Y-%m-%d")

    st.dataframe(
        confidence_df[
            [
                "published",
                "source",
                "title",
                "sentiment_label",
                "positive",
                "neutral",
                "negative",
                "confidence",
                "relevance_score",
                "extraction_method",
                "url",
            ]
        ].sort_values("confidence", ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config={
            "url": st.column_config.LinkColumn("Article"),
            "positive": st.column_config.NumberColumn("Positive", format="%.3f"),
            "neutral": st.column_config.NumberColumn("Neutral", format="%.3f"),
            "negative": st.column_config.NumberColumn("Negative", format="%.3f"),
            "confidence": st.column_config.ProgressColumn(
                "Confidence", min_value=0, max_value=1, format="%.2f"
            ),
            "relevance_score": st.column_config.NumberColumn("Relevance"),
        },
    )

with methodology_tab:
    st.markdown(
        """
        ### How this dashboard works

        1. **Search:** DDGS searches recent news for the exact company name and optional industry qualifier.
        2. **Clean:** Tracking parameters, excluded domains, duplicate URLs, and results outside the selected date period are removed.
        3. **Extract:** Trafilatura attempts to extract readable article content. If extraction fails, the DDGS title and snippet are used.
        4. **Validate:** Exact company-name mentions generate a relevance score. Low-relevance results are discarded.
        5. **Analyze:** FinBERT scores company-focused article context as positive, neutral, or negative.
        6. **Cache:** Search results, extracted content, and sentiment probabilities are stored in SQLite.

        ### Important limitations

        - DDGS is a search-results provider, not a complete media archive.
        - Paywalls, blocked pages, missing dates, and unavailable URLs can reduce coverage.
        - Ambiguous company names should include an industry qualifier.
        - The media reputation index summarizes analyzed coverage only. It is not a definitive measure of company quality or public opinion.
        """
    )
