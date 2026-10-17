# Experience-Curve Calculator

Free Streamlit app powered by the SEC EDGAR Company Facts API. Enter an SEC stock ticker, retrieve five or more years of original 10-K revenue and cost figures, supply sourced unit volumes or annual average selling prices, and compute a log-log experience curve.

## Publish for free
1. Sign up for GitHub and Streamlit Community Cloud.
2. Create a **public** GitHub repository, e.g. `learning-curve-calculator`.
3. Upload `app.py` and `requirements.txt` to its root.
4. Visit https://share.streamlit.io, choose **Create app**, and select the repository and `app.py` as the entrypoint.
5. Deploy, copy the app URL, and test the public app in a private/incognito browser window.

## Use
- Enter a real contact email in the sidebar. The SEC requires a descriptive User-Agent; the address is transmitted to SEC endpoints, so use an address you're comfortable providing. Never place it directly in code.
- Enter e.g. `GME` and choose a fiscal-year count (5–15).
- Click **Retrieve SEC 10-K data**. Inspect revenue, COGS, fiscal dates and original filing links.
- Select direct units or revenue/average-selling-price. Enter positive values for *each* fiscal year in the editable table. Provide a defensible written source/method note.
- Click **Calculate experience curve**. Use the resulting metrics, chart, SEC filings and methodology in your assignment PDF. You can download CSV and report text.

## Key limitation
EDGAR's standard company facts reliably provides some financial accounting concepts but not standardized unit volumes for arbitrary companies. This tool deliberately **does not fabricate sales units or selling prices**. The app retrieves raw costs and revenue; the researcher supplies sourced unit volumes or justified annual ASP estimates before a curve can be fit. Some tickers use custom tags and require manual source review. R² measures fit in logarithmic space and should not be interpreted as causal proof of learning.
