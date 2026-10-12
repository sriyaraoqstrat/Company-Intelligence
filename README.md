# Company Reputation Intelligence Dashboard

## Setup on Windows PowerShell

```powershell
cd path\to\company_reputation_dashboard
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The first sentiment run downloads `ProsusAI/finbert` from Hugging Face. The SQLite database is created automatically as `reputation_cache.db`.

## Files
- `app.py`: interactive Streamlit dashboard
- `pipeline.py`: DDGS search, Trafilatura extraction, relevance filtering, FinBERT inference
- `database.py`: SQLite schema and cache operations
- `requirements.txt`: dependencies

## Notes
- Use an industry qualifier for ambiguous company names.
- Select **Refresh from internet** to replace cached data.
- DDGS coverage is not exhaustive. Respect publishers' terms and do not attempt to bypass paywalls.
