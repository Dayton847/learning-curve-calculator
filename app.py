"""Public company experience-curve calculator; SEC EDGAR, user-supplied units/proxies."""
import io
import math
import re
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="Experience-Curve Calculator", page_icon="📉", layout="wide")
st.title("Experience-Curve Calculator")
st.caption("SEC EDGAR annual 10-K data · transparent unit assumptions · log-log power regression")

API = "https://data.sec.gov"
TICKERS = "https://www.sec.gov/files/company_tickers.json"
COST_TAGS = ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold", "CostOfSales", "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization", "CostOfServices"]
REVENUE_TAGS = ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueGoodsNet"]


def sec_get(url, contact):
    if not contact.strip() or "@" not in contact:
        raise ValueError("Enter an email address for the SEC's required User-Agent contact information.")
    response = requests.get(url, headers={"User-Agent": f"ExperienceCurveResearch student-research {contact.strip()}", "Accept-Encoding": "gzip, deflate", "Host": requests.utils.urlparse(url).netloc}, timeout=30)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=86400, show_spinner=False)
def ticker_to_cik(ticker, contact):
    data = sec_get(TICKERS, contact)
    matches = [v for v in data.values() if v.get("ticker", "").upper() == ticker.upper()]
    if not matches:
        raise ValueError(f"No matching SEC ticker found for {ticker}.")
    return int(matches[0]["cik_str"]), matches[0]["title"]


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_facts(cik, contact):
    return sec_get(f"{API}/api/xbrl/companyfacts/CIK{cik:010d}.json", contact)


def annual_observations(facts, tags):
    """Original annual 10-K periods only; no mixing instant figures or quarters."""
    items = []
    gaap = facts.get("facts", {}).get("us-gaap", {})
    for tag in tags:
        for entry in gaap.get(tag, {}).get("units", {}).get("USD", []):
            if entry.get("form") != "10-K" or entry.get("fp") != "FY":
                continue
            if not all(k in entry for k in ("start", "end", "val", "accn", "filed")):
                continue
            try:
                duration = (pd.Timestamp(entry["end"]) - pd.Timestamp(entry["start"])).days
            except (ValueError, TypeError):
                continue
            if not 330 <= duration <= 380 or entry["val"] <= 0:
                continue
            items.append({"end": entry["end"], "start": entry["start"], "amount": float(entry["val"]), "tag": tag, "accn": entry["accn"], "filed": entry["filed"], "fy": entry.get("fy")})
    # Per period prefer earliest original filing, then tag priority when same filing.
    rank = {t: i for i, t in enumerate(tags)}
    items.sort(key=lambda r: (r["end"], r["filed"], rank[r["tag"]]))
    out = {}
    for r in items:
        out.setdefault(r["end"], r)
    return out


def filing_url(cik, accn):
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{accn.replace('-', '')}/{accn}-index.htm"


def build_financials(facts, cik, count):
    revenues = annual_observations(facts, REVENUE_TAGS)
    costs = annual_observations(facts, COST_TAGS)
    rows = []
    for end in sorted(set(revenues) & set(costs)):
        rev, cost = revenues[end], costs[end]
        # Demand same annual period and same 10-K to avoid inconsistent comparative figures.
        if rev["start"] != cost["start"] or rev["accn"] != cost["accn"]:
            continue
        rows.append({"Fiscal year ended": end, "Revenue ($)": rev["amount"], "COGS ($)": cost["amount"], "Revenue tag": rev["tag"], "COGS tag": cost["tag"], "10-K filed": cost["filed"], "SEC accession": cost["accn"], "10-K URL": filing_url(cik, cost["accn"])})
    return pd.DataFrame(rows[-count:]).reset_index(drop=True)


def calculate(df, mode):
    work = df.copy()
    work["Revenue ($)"] = pd.to_numeric(work["Revenue ($)"], errors="coerce")
    work["COGS ($)"] = pd.to_numeric(work["COGS ($)"], errors="coerce")
    if mode == "Direct reported units":
        units = pd.to_numeric(work["Units sold"], errors="coerce")
    else:
        asp = pd.to_numeric(work["Average selling price ($/unit)"], errors="coerce")
        if asp.isna().any() or (asp <= 0).any():
            raise ValueError("Provide a positive average selling price for every fiscal year.")
        units = work["Revenue ($)"] / asp
    if units.isna().any() or (units <= 0).any():
        raise ValueError("Provide positive units for every fiscal year.")
    if len(work) < 5:
        raise ValueError("At least five annual observations are required.")
    work["Annual units"] = units.astype(float)
    work["Cumulative units"] = work["Annual units"].cumsum()
    work["Unit cost ($)"] = work["COGS ($)"] / work["Annual units"]
    if not np.isfinite(work[["Cumulative units", "Unit cost ($)"]].to_numpy(dtype=float)).all():
        raise ValueError("The inputs generate non-finite values. Check unit assumptions.")
    x = np.log(work["Cumulative units"].to_numpy(dtype=float))
    y = np.log(work["Unit cost ($)"] .to_numpy(dtype=float))
    if np.ptp(x) < 1e-12:
        raise ValueError("Cumulative output has insufficient variation.")
    slope, intercept = np.polyfit(x, y, 1)
    yhat = intercept + slope * x
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - float(np.sum((y - yhat) ** 2)) / sst if sst > 1e-20 else float('nan')
    b = -float(slope)
    learning_rate = 1 - 2 ** (-b)
    return work, float(np.exp(intercept)), b, learning_rate, r2


def curve_chart(work, a, b):
    xx = np.geomspace(float(work["Cumulative units"].min()), float(work["Cumulative units"].max()), 250)
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=work["Cumulative units"], y=work["Unit cost ($)"], mode="markers+text", text=work["Fiscal year ended"], textposition="top center", name="Observed", marker={"size": 10}))
    figure.add_trace(go.Scatter(x=xx, y=a * xx ** (-b), name="Fitted power curve", mode="lines"))
    figure.update_layout(xaxis={"type": "log", "title": "Cumulative units since FIRST year shown (log scale)"}, yaxis={"type": "log", "title": "COGS / annual units ($ per unit, log scale)"}, margin={"l": 20, "r": 20, "t": 20, "b": 30}, height=450)
    return figure


def interpretation(rate, r2, company):
    fit = ("strong" if r2 >= 0.8 else "moderate" if r2 >= 0.5 else "weak") if np.isfinite(r2) else "undefined"
    trend = (f"Estimated unit cost declines {rate:.1%} per doubling" if rate > 0 else f"Estimated unit cost increases {abs(rate):.1%} per doubling")
    msg = f"For {company}, {trend} of cumulative units within the selected window. The log-log regression has R² = {r2:.3f}, which is a {fit} fit under the assignment's approximate thresholds. "
    if not np.isfinite(r2) or r2 < .5:
        msg += "This window does not provide convincing evidence of an experience-curve advantage. "
    elif rate <= 0:
        msg += "Despite the fitted relationship, unit costs do not fall with greater cumulative volume, so no cost-based learning advantage is apparent. "
    else:
        msg += "This is consistent with a scale-related cost advantage, not proof that cumulative experience caused it. "
    msg += "Changes in product mix, pricing, sourcing, and accounting can affect the measured relationship."
    return msg


with st.sidebar:
    st.subheader("SEC access")
    contact = st.text_input("Contact email (required by SEC)", placeholder="your-email@example.com", help="Sent in the SEC request User-Agent. Not stored persistently by this app.")
    ticker = st.text_input("US stock ticker", value="GME").strip().upper()
    years = st.slider("Annual years to retrieve", 5, 15, 7)
    if "loaded_key" not in st.session_state:
        st.session_state.loaded_key = None
    load = st.button("Retrieve SEC 10-K data", type="primary", use_container_width=True)

st.info("This tool can automatically retrieve standardized financial statement figures. **Annual units sold are NOT reliably present in SEC XBRL data.** Add disclosed units or defensible annual average selling prices below; the calculator will never invent them.")

if load:
    try:
        if not ticker or not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,9}", ticker):
            raise ValueError("Enter a valid US stock ticker.")
        with st.spinner("Fetching the SEC ticker map and annual company facts..."):
            cik, company = ticker_to_cik(ticker, contact)
            facts = fetch_facts(cik, contact)
            financials = build_financials(facts, cik, years)
        if len(financials) < 5:
            raise ValueError(f"Only {len(financials)} matching annual observations were found. This company may use custom XBRL tags or have insufficient SEC data. Choose another ticker or verify filings manually.")
        st.session_state["financials"] = financials
        st.session_state["company"] = company
        st.session_state["cik"] = cik
        st.session_state["loaded_key"] = (ticker, years)
    except (requests.RequestException, ValueError, KeyError) as err:
        st.error(f"Could not retrieve usable data: {err}")

if "financials" not in st.session_state:
    st.write("Enter your contact email, then click **Retrieve SEC 10-K data** to begin.")
    st.stop()

base = st.session_state["financials"].copy()
company = st.session_state["company"]
st.subheader(f"{company} — {len(base)} annual 10-K periods")
st.caption("Values come from SEC companyfacts and are linked to the specific accession numbers below. The tool uses a consistent original-filing annual period for each observation; it does not splice quarterly and full-year data.")
st.dataframe(base[["Fiscal year ended", "Revenue ($)", "COGS ($)", "10-K filed", "SEC accession", "10-K URL", "Revenue tag", "COGS tag"]], use_container_width=True, hide_index=True, column_config={"10-K URL": st.column_config.LinkColumn("Original 10-K filing")})

mode = st.radio("How will you estimate annual units?", ["Revenue / documented average selling price", "Direct reported units"], horizontal=True)
source = st.text_area("Unit data source and method (required)", placeholder="For each year, cite the disclosed sales units or average price source and describe what is included. If an analyst estimate, explain and defend the assumptions.", height=110)
input_table = base[["Fiscal year ended", "Revenue ($)", "COGS ($)"]].copy()
if mode == "Direct reported units":
    input_table["Units sold"] = [None] * len(input_table)
    edited = st.data_editor(input_table, hide_index=True, use_container_width=True, disabled=["Fiscal year ended", "Revenue ($)", "COGS ($)"], num_rows="fixed", key="unit_direct")
else:
    input_table["Average selling price ($/unit)"] = [None] * len(input_table)
    edited = st.data_editor(input_table, hide_index=True, use_container_width=True, disabled=["Fiscal year ended", "Revenue ($)", "COGS ($)"], num_rows="fixed", key="unit_asp")

if st.button("Calculate experience curve", type="primary"):
    try:
        if len(source.strip()) < 25:
            raise ValueError("Describe the source and justification of the unit estimates (at least 25 characters).")
        work, a, b, rate, r2 = calculate(edited, mode)
        st.session_state["result"] = {"work": work, "a": a, "b": b, "rate": rate, "r2": r2, "mode": mode, "source": source, "company": company, "base": base}
    except (ValueError, TypeError, np.linalg.LinAlgError) as err:
        st.error(str(err))

result = st.session_state.get("result")
if result and result["company"] == company:
    st.divider()
    st.subheader("Calculated experience curve")
    c1, c2, c3 = st.columns(3)
    c1.metric("Learning rate per doubling", f"{result['rate']:.2%}")
    c2.metric("Coefficient b", f"{result['b']:.4f}")
    c3.metric("Log-log R²", f"{result['r2']:.4f}")
    st.latex(r"Cost(N)=a\,N^{-b}\qquad Learning\ Rate=1-2^{-b}")
    st.write(f"Fitted curve: **Unit cost = {result['a']:.6g} × (cumulative units)^(-{result['b']:.5f})**")
    st.plotly_chart(curve_chart(result["work"], result["a"], result["b"]), use_container_width=True)
    st.write(interpretation(result["rate"], result["r2"], company))
    st.warning("Cumulative units start at zero BEFORE the first fiscal year in THIS dataset; the first plotted cumulative value equals that year's annual units. This is not lifetime company output. For retailers, COGS/units primarily reflects merchandise cost and mix, not production learning. Correlation is not causation.")
    st.markdown(f"**Unit method:** {result['mode']}  \n**Source and justification:** {result['source']}")
    st.dataframe(result["work"], use_container_width=True, hide_index=True)
    st.download_button("Download calculations and filing references (CSV)", data=result["work"].to_csv(index=False).encode("utf-8"), file_name=f"{ticker.lower()}_experience_curve.csv", mime="text/csv")
    report = f"{company} ({ticker}) experience-curve output\nLearning rate: {result['rate']:.4%}\nb: {result['b']:.6f}\nR²: {result['r2']:.5f}\nRegression: Cost(N) = {result['a']:.8g} * N^(-{result['b']:.6f})\nMethod: {result['mode']}\nUnit source: {result['source']}\nCumulative units start with the first year in the selected window, not lifetime production.\n\n" + interpretation(result['rate'], result['r2'], company) + "\n\nSEC filings:\n" + "\n".join(f"{r['Fiscal year ended']}: {r['10-K URL']} (filed {r['10-K filed']})" for _, r in result["base"].iterrows())
    st.download_button("Download report text", data=report.encode("utf-8"), file_name=f"{ticker.lower()}_report.txt", mime="text/plain")

with st.expander("Method and caveats"):
    st.markdown("""- SEC ticker list: https://www.sec.gov/files/company_tickers.json; SEC annual company facts: https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json
- Original Form 10-K annual observations are selected by matching revenue and COGS on exact beginning/end dates and accession numbers. Fiscal periods are 330–380 days to allow 52/53-week years.
- Model is fitted using ordinary least squares on **natural log(unit cost)** versus **natural log(cumulative estimated units)**. The model is `ln(cost) = ln(a) - b ln(N)`, and the doubling-based cost reduction is `1 - 2**(-b)`.
- R² is evaluated in **log space**. Approximate classroom interpretation: above 0.8 supports a strong fitted relationship; below 0.5 means no clear curve in the window.
- Units are never extracted automatically from text-based disclosures; the user must enter and cite them. Revenue ÷ ASP captures mix and price shifts, and different products may not represent comparable units.
- Some companies use custom taxonomy tags; for them this general-purpose automatic extractor may not find five valid annual observations. Check each original SEC filing.
- Do not make capital-investment or pricing decisions solely from these observational regressions.
""")
