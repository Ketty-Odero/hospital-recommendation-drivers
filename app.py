"""
What Makes Patients Recommend a Hospital?
Streamlit app for the HCAHPS predictive analytics project.
Rebecca Quaye · Ketty Odero · Victoria Kutuusa

Run:   python train_models.py   (once, saves the trained models)
       streamlit run app.py
"""

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score)

from train_models import (DATA_FILE, HIGH, MODEL_FILE, PREDICTORS, TARGET, THRESHOLD, TIER_ORDER,
                          load_data, train_all)

st.set_page_config(page_title="What Makes Patients Recommend a Hospital?", page_icon="🏥", layout="wide")

# Palette shared with the slides and report
NAVY, TEAL, TEAL_DARK, STEEL, CORAL, MIST = "#0B2540", "#0E7C7B", "#0B5E5D", "#4F7FA8", "#C9562F", "#9DB7CC"
PAPER, TINT, CARD, INK, BODY, MUTED = "#F6F9FA", "#E6F0F2", "#FDFEFE", "#12304D", "#3A4E63", "#5B6B7B"
TIER_COLORS = {"Higher": TEAL, "Moderate": STEEL, "Lower": CORAL}
LABELS = {
    "nurse_communication": "Nurse communication", "doctor_communication": "Doctor communication",
    "medicine_explanation": "Medicine explanation", "discharge_information": "Discharge information",
    "room_cleanliness": "Room cleanliness", "night_quietness": "Night quietness",
}
SHORT = {"nurse_communication": "Nurse", "doctor_communication": "Doctor", "medicine_explanation": "Medicines",
         "discharge_information": "Discharge", "room_cleanliness": "Cleanliness", "night_quietness": "Night quiet"}

# ───────────────────────────────────────────── look and feel
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=Inter:wght@400;500;600;700;800&display=swap');
html, body, .stApp, [data-testid="stMarkdownContainer"], button, input, label {{ font-family: Inter, Arial, sans-serif; }}
.stApp {{ background: {PAPER}; }}
.block-container {{ padding-top: 4.5rem; max-width: 1240px; }}
[data-testid="stSidebar"] {{ background: {NAVY}; }}
[data-testid="stSidebar"] * {{ color: #CFE0EA; }}
[data-testid="stSidebarNav"] a {{ border-radius: 10px; }}
[data-testid="stSidebarNav"] a[aria-current="page"] {{ background: #143A5E; border-left: 4px solid {TEAL}; }}
[data-testid="stSidebarNav"] a[aria-current="page"] span {{ color: #F6F9FA; font-weight: 600; }}
[data-testid="stNavSectionHeader"], [data-testid="stNavSectionHeader"] * {{ color: {MIST} !important; text-transform: uppercase; letter-spacing: 1.2px; font-size: 11px; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ background: {CARD}; border-radius: 18px; border: none !important;
    box-shadow: 0 4px 18px rgba(11,37,64,0.07); }}
.kicker {{ font-size: 13px; font-weight: 700; letter-spacing: 1.4px; text-transform: uppercase; color: {TEAL}; margin-bottom: 4px; }}
h1.page-title {{ font-family: 'DM Serif Display', Georgia, serif !important; font-weight: 400 !important; font-size: 42px !important;
    line-height: 1.1 !important; color: {NAVY} !important; margin: 0 0 8px 0 !important; padding: 0 !important; letter-spacing: 0 !important; }}
.lead {{ font-size: 16px; color: {BODY}; line-height: 1.55; max-width: 880px; margin: 0 0 6px 0; }}
.question {{ font-size: 15px; font-weight: 600; color: {STEEL}; margin: 4px 0 2px; }}
.kpis {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 16px; margin: 6px 0 4px; }}
.kpi {{ background: {CARD}; border-radius: 16px; padding: 18px 20px; box-shadow: 0 4px 18px rgba(11,37,64,0.07);
    display: flex; flex-direction: column; gap: 6px; }}
.kpi .lab {{ font-size: 13px; font-weight: 600; color: {MUTED}; }}
.kpi .val {{ font-family: 'DM Serif Display', Georgia, serif; font-size: 30px; line-height: 1.1; color: {NAVY};
    word-break: normal; overflow-wrap: normal; hyphens: none; }}
.kpi .sub {{ font-size: 13px; color: {MUTED}; line-height: 1.45; }}
.kpi.accent {{ background: {TEAL}; }}
.kpi.accent .lab, .kpi.accent .sub {{ color: #CDEBE6; }}
.kpi.accent .val {{ color: {PAPER}; }}
.box {{ border-radius: 16px; padding: 18px 22px; margin-bottom: 12px; }}
.box h4 {{ margin: 0 0 8px; font-size: 16px; font-weight: 700; color: {NAVY}; }}
.box p {{ margin: 0; font-size: 14px; line-height: 1.55; color: {BODY}; }}
.box.tint {{ background: {TINT}; }}
.box.white {{ background: {CARD}; box-shadow: 0 4px 18px rgba(11,37,64,0.07); }}
.box.dark {{ background: {NAVY}; }}
.box.dark h4 {{ color: {PAPER}; }}
.box.dark p, .box.dark li {{ color: #CFE0EA; }}
.box.warn {{ background: #FCEFD9; }}
.box.warn p {{ color: #6B4100; }}
.box.risk {{ background: #F6DED4; }}
.box.risk p {{ color: #7A2E14; }}
.box.good {{ background: #CDEBE6; }}
.box.good p {{ color: {TEAL_DARK}; }}
.chart-title {{ font-size: 17px; font-weight: 700; color: {NAVY}; margin: 0; }}
.chart-sub {{ font-size: 13px; color: {MUTED}; margin: 2px 0 0; line-height: 1.45; }}
.step {{ display: flex; gap: 14px; align-items: flex-start; margin-bottom: 14px; }}
.step .n {{ flex: none; width: 32px; height: 32px; border-radius: 16px; background: {NAVY}; color: {PAPER}; font-weight: 700;
    display: flex; align-items: center; justify-content: center; }}
.step b {{ display: block; font-size: 15px; color: {INK}; }}
.step span {{ font-size: 14px; color: {BODY}; line-height: 1.45; }}
.result {{ border-radius: 16px; padding: 16px 22px; display: flex; align-items: center; gap: 20px; margin-bottom: 12px; }}
.result .txt {{ flex: 1; }}
.result .txt b {{ display: block; font-size: 15px; margin-bottom: 3px; }}
.result .txt span {{ font-size: 13px; line-height: 1.45; }}
.result .num {{ text-align: right; flex: none; }}
.result .num .big {{ font-family: 'DM Serif Display', Georgia, serif; font-size: 38px; line-height: 1; display: block; }}
.result .num small {{ font-size: 13px; font-weight: 600; }}
.bars .row {{ display: grid; grid-template-columns: minmax(110px, 0.9fr) minmax(90px, 1.6fr) auto; align-items: center;
    column-gap: 14px; margin: 8px 0; font-size: 14px; color: {INK}; }}
.bars .name {{ min-width: 0; }}
.bars .track {{ height: 20px; background: #EEF3F6; border-radius: 6px; overflow: hidden; }}
.bars .fill {{ height: 20px; border-radius: 6px; }}
.bars .val {{ font-weight: 700; white-space: nowrap; }}
.driver {{ display: flex; gap: 16px; align-items: flex-start; padding: 14px 0; border-bottom: 1px solid {TINT}; }}
.driver:last-child {{ border-bottom: none; }}
.driver .rank {{ flex: none; width: 36px; height: 36px; border-radius: 18px; color: {PAPER}; font-weight: 800;
    display: flex; align-items: center; justify-content: center; }}
.driver .body {{ flex: 1; }}
.driver .top {{ display: flex; justify-content: space-between; gap: 12px; align-items: center; }}
.driver .top b {{ font-size: 16px; color: {NAVY}; }}
.tag {{ font-size: 12px; font-weight: 700; padding: 3px 9px; border-radius: 6px; white-space: nowrap; }}
.driver p {{ margin: 4px 0 0; font-size: 14px; color: {BODY}; line-height: 1.45; }}
.driver .act {{ color: {TEAL}; font-weight: 600; }}
.grid-rank {{ display: grid; grid-template-columns: 1.6fr repeat(5, 1fr); gap: 7px 4px; font-size: 12px; align-items: center; }}
.grid-rank .h {{ color: {MUTED}; font-weight: 600; text-align: center; line-height: 1.2; }}
.grid-rank .r {{ justify-self: center; width: 28px; height: 28px; border-radius: 14px; font-weight: 800;
    display: flex; align-items: center; justify-content: center; }}
.mtab {{ display: grid; font-size: 14px; }}
.mtab span {{ padding: 8px 0; border-bottom: 1px solid {TINT}; color: {INK}; }}
.mtab .hd {{ color: {MUTED}; font-weight: 600; border-bottom: 1.5px solid #D5E2E7; }}
.mtab .best {{ color: {TEAL}; font-weight: 700; }}
.mtab .mean {{ color: {BODY}; }}
.cm {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }}
.cm div {{ border-radius: 12px; padding: 12px 14px; }}
.cm .n {{ font-size: 24px; font-weight: 800; display: block; }}
.cm b {{ font-size: 13px; display: block; }}
.cm small {{ font-size: 12px; line-height: 1.4; }}
.foot {{ font-size: 12px; color: {MUTED}; margin-top: 8px; }}
div.stButton > button[kind="primary"], [data-testid="stPageLink"] a {{ border-radius: 10px; }}
</style>
""", unsafe_allow_html=True)


def header(kicker, title, lead=None, color=TEAL):
    # Kicker, serif title and optional one-line explanation
    st.markdown(f'<div class="kicker" style="color:{color}">{kicker}</div><h1 class="page-title">{title}</h1>'
                + (f'<p class="lead">{lead}</p>' if lead else ""), unsafe_allow_html=True)


def box(title, text, kind="white"):
    # A coloured card with a title and a paragraph
    st.markdown(f'<div class="box {kind}">' + (f"<h4>{title}</h4>" if title else "") + f"<p>{text}</p></div>",
                unsafe_allow_html=True)


def chart_title(title, sub=None):
    st.markdown(f'<p class="chart-title">{title}</p>' + (f'<p class="chart-sub">{sub}</p>' if sub else ""),
                unsafe_allow_html=True)


def style(fig, height=380, legend=True):
    # One plotly look for every chart
    fig.update_layout(height=height, margin=dict(l=8, r=16, t=16, b=8), plot_bgcolor=CARD, paper_bgcolor=CARD,
                      font=dict(family="Inter, Arial, sans-serif", size=13, color=INK), showlegend=legend,
                      legend=dict(orientation="h", y=1.08, x=0, title=None, font=dict(size=13)),
                      hoverlabel=dict(font=dict(family="Inter, Arial, sans-serif")))
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#D5E2E7", tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor="#EEF3F6", zeroline=False, tickfont=dict(color=INK))
    return fig


def bars_html(rows, fmt_color=None):
    # Simple horizontal bars: rows = [(name, width 0-100, value text, colour)]
    out = '<div class="bars">'
    for name, width, text, color in rows:
        out += (f'<div class="row"><span class="name">{name}</span><div class="track"><div class="fill" '
                f'style="width:{max(width, 1):.0f}%;background:{color}"></div></div>'
                f'<span class="val" style="color:{color if color in (TEAL, CORAL) else INK}">{text}</span></div>')
    return out + "</div>"


# ───────────────────────────────────────────── data and models
@st.cache_data(show_spinner=False)
def get_data():
    return load_data(DATA_FILE)


@st.cache_resource(show_spinner="Loading the trained models…")
def get_models(_df):
    # Load the saved models; if the file is missing or from another library version, train once and save
    try:
        return joblib.load(MODEL_FILE), "saved"
    except Exception:
        bundle = train_all(_df)
        try:
            MODEL_FILE.parent.mkdir(exist_ok=True)
            joblib.dump(bundle, MODEL_FILE)
        except OSError:
            pass
        return bundle, "trained now"


def tier_of(M, X):
    return [M["tier_names"][int(k)] for k in M["kmeans"].predict(M["pca"].transform(M["pca_scaler"].transform(X[PREDICTORS])))]


def evaluate(M, reg_df, cls_df):
    # Score every model on the given hospitals (regression and classification sets may differ)
    reg, cls, cms = {}, {}, {}
    if reg_df is not None and len(reg_df):
        Xr, yr = reg_df[PREDICTORS], reg_df[TARGET]
        for name, m in [("Linear regression", M["linear"]), ("Decision tree", M["tree_reg"])]:
            p = m.predict(Xr)
            reg[name] = {"R²": r2_score(yr, p), "MAE": mean_absolute_error(yr, p),
                         "RMSE": float(np.sqrt(mean_squared_error(yr, p))),
                         "within10": float((np.abs(p - yr) <= 10).mean())}
    if cls_df is not None and len(cls_df):
        Xc, yc = cls_df[PREDICTORS], (cls_df[TARGET] >= THRESHOLD).astype(int)
        for name, m in [("Logistic regression", M["logistic"]), ("Classification tree", M["class_tree"]),
                        ("Random forest", M["forest"])]:
            pred, prob = m.predict(Xc), m.predict_proba(Xc)[:, 1]
            cls[name] = {"Accuracy": accuracy_score(yc, pred), "Precision": precision_score(yc, pred, zero_division=0),
                         "Recall": recall_score(yc, pred, zero_division=0), "F1": f1_score(yc, pred, zero_division=0),
                         "ROC-AUC": roc_auc_score(yc, prob) if yc.nunique() > 1 else np.nan}
            cms[name] = confusion_matrix(yc, pred, labels=[0, 1])
        cls["_baseline"] = max(yc.mean(), 1 - yc.mean())
    return reg, cls, cms


def score_hospitals(M, frame):
    # Predicted %, chance of clearing 70%, peer tier and a plain flag for each row
    out = frame.copy()
    out["predicted_recommend_pct"] = np.clip(M["linear"].predict(frame[PREDICTORS]), 0, 100).round(1)
    out["chance_70_or_more"] = M["logistic"].predict_proba(frame[PREDICTORS])[:, 1].round(3)
    out["peer_tier"] = tier_of(M, frame)
    out["flag"] = np.select([out["chance_70_or_more"] < 0.5, out["chance_70_or_more"] < 0.65],
                            ["At risk", "Watch"], "OK")
    return out


df = get_data()
M, model_source = get_models(df)
N = len(df)
id_col = df["facility_id"] if "facility_id" in df.columns else pd.Series(df.index.astype(str))
reg_test = df[id_col.isin(M["reg_test_ids"])]
cls_test = df[id_col.isin(M["cls_test_ids"])]
REG, CLS, CMS = evaluate(M, reg_test, cls_test)
MEANS = df[PREDICTORS].mean()
HI, LO = df[df[HIGH] == 1], df[df[HIGH] == 0]


def hospital_labels(frame):
    if "facility_name" not in frame.columns:
        return frame.index.astype(str)
    lab = frame["facility_name"].astype(str).str.title()
    if {"city", "state"} <= set(frame.columns):
        lab = lab + " (" + frame["city"].astype(str).str.title() + ", " + frame["state"].astype(str) + ")"
    return lab


# ───────────────────────────────────────────── pages
def page_overview():
    header("The problem", "Why do patients recommend one hospital and not another?",
           f"Same national survey, same questions, yet some hospitals are recommended three times as often as others. "
           f"We used {N:,} hospitals to find which parts of a stay matter most, and built a tool that predicts where a hospital will land.")
    coef = pd.Series(M["linear"].coef_, index=PREDICTORS)
    top = coef.idxmax()
    st.markdown(f"""
<div class="kpis">
<div class="kpi"><span class="lab">Hospitals in our study</span><span class="val">{N:,}</span>
<span class="sub">US hospitals with at least 100 patient surveys</span></div>
<div class="kpi"><span class="lab">Middle recommendation rate</span><span class="val">{df[TARGET].median():.0f}%</span>
<span class="sub">Half of hospitals are above this share of patients who would definitely recommend, half below</span></div>
<div class="kpi"><span class="lab">Highest vs lowest hospital</span><span class="val">{df[TARGET].max():.0f}% vs {df[TARGET].min():.0f}%</span>
<span class="sub">Share of each one’s patients who would definitely recommend it</span></div>
<div class="kpi accent"><span class="lab">Strongest driver of recommendation</span><span class="val">{LABELS[top]}</span>
<span class="sub">Each extra point here lifts recommendation by about {coef[top]:.1f} of a point</span></div>
</div>""", unsafe_allow_html=True)
    st.write("")
    left, right = st.columns([1.35, 1], gap="medium")
    with left:
        with st.container(border=True):
            chart_title("Most hospitals sit near 70%, with a long tail of low performers",
                        "Each bar counts the hospitals at that recommendation level. Taller bar = more hospitals.")
            fig = go.Figure()
            fig.add_histogram(x=LO[TARGET], xbins=dict(start=29.5, end=95.5, size=2), marker_color=CORAL,
                              name=f"{len(LO):,} hospitals below 70%")
            fig.add_histogram(x=HI[TARGET], xbins=dict(start=29.5, end=95.5, size=2), marker_color=TEAL,
                              name=f"{len(HI):,} hospitals at 70% or more")
            fig.update_layout(barmode="overlay", bargap=0.08)
            fig.update_xaxes(title="% of a hospital’s patients who would definitely recommend it", ticksuffix="%")
            fig.update_yaxes(title="Hospitals")
            st.plotly_chart(style(fig, 330), width="stretch")
    with right:
        st.markdown(f"""<div class="box tint"><h4>What this app adds</h4>
<div class="step"><span class="n">1</span><div><b>Rank what matters</b><span>Which parts of the stay are most linked to recommendation</span></div></div>
<div class="step"><span class="n">2</span><div><b>Compare with peers</b><span>Each hospital against its own quality tier</span></div></div>
<div class="step"><span class="n">3</span><div><b>Flag risk and test plans</b><span>Spot hospitals likely below 70% and try what-if changes</span></div></div>
</div>""", unsafe_allow_html=True)
        st.page_link(P_PREDICT, label="Try the hospital predictor", icon=":material/arrow_forward:")
        st.page_link(P_DRIVERS, label="See what drives recommendation", icon=":material/insights:")
    st.markdown('<p class="foot">Main question: which patient-experience dimensions matter most for whether patients recommend a hospital?</p>',
                unsafe_allow_html=True)


def page_explore():
    header("Exploratory data analysis", "What the data shows before any modelling", color=STEEL)
    t1, t2, t3, t4 = st.tabs(["Best and worst scores", "Top hospitals vs the rest", "Region and survey volume",
                              "Correlations and outliers"])

    with t1:
        st.markdown('<p class="question">Which aspects of hospital stays receive the highest and lowest scores?</p>', unsafe_allow_html=True)
        s = pd.DataFrame({"mean": MEANS, "p10": df[PREDICTORS].quantile(.1), "p90": df[PREDICTORS].quantile(.9),
                          "sd": df[PREDICTORS].std()}).sort_values("mean")
        left, right = st.columns([1.6, 1], gap="medium")
        with left, st.container(border=True):
            chart_title("Hospitals do best at discharge information and worst at night-time quiet",
                        "Dot = national average. Band = from a weaker (10th percentile) to a stronger (90th percentile) hospital.")
            fig = go.Figure()
            names = [LABELS[c] for c in s.index]
            for name, (_, r) in zip(names, s.iterrows()):
                fig.add_trace(go.Scatter(x=[r.p10, r.p90], y=[name, name], mode="lines", line=dict(color="#CDEBE6", width=16),
                                         showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=s["mean"], y=names, mode="markers+text", text=[f"{v:.0f}%" for v in s["mean"]],
                                     textposition="top center", marker=dict(size=16, color=[CORAL, CORAL] + [TEAL] * 4),
                                     showlegend=False, hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
            fig.update_xaxes(title="% of patients giving the best answer", range=[40, 100], ticksuffix="%")
            st.plotly_chart(style(fig, 360, legend=False), width="stretch")
        with right:
            box("Best", f"{LABELS[s.index[-1]]}, {s['mean'].iloc[-1]:.0f}%: most patients feel prepared to go home.", "good")
            box("Weakest", f"{LABELS[s.index[0]]} ({s['mean'].iloc[0]:.0f}%) and {LABELS[s.index[1]].lower()} ({s['mean'].iloc[1]:.0f}%).", "risk")
            most = s["sd"].idxmax()
            box("Where hospitals differ most",
                f"{LABELS[most]}: a typical hospital is about {s.loc[most, 'sd']:.0f} points from the average, so there are better hospitals to learn from.", "tint")

    with t2:
        st.markdown('<p class="question">How do high-recommendation hospitals differ from lower-recommendation hospitals?</p>', unsafe_allow_html=True)
        g = pd.DataFrame({"high": HI[PREDICTORS].mean(), "low": LO[PREDICTORS].mean()})
        g["gap"] = g["high"] - g["low"]
        pooled = np.sqrt(((len(HI) - 1) * HI[PREDICTORS].var() + (len(LO) - 1) * LO[PREDICTORS].var()) / (len(HI) + len(LO) - 2))
        g["d"] = g["gap"] / pooled
        g = g.sort_values("gap")
        left, right = st.columns([1.6, 1], gap="medium")
        with left, st.container(border=True):
            chart_title("Top hospitals lead on every part of the stay",
                        "Average score of hospitals at 70% or more (teal) and below 70% (orange). The number is how many points ahead.")
            names = [LABELS[c] for c in g.index]
            fig = go.Figure()
            for name, (_, r) in zip(names, g.iterrows()):
                fig.add_trace(go.Scatter(x=[r.low, r.high], y=[name, name], mode="lines", line=dict(color="#D5E2E7", width=5),
                                         showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=g["low"], y=names, mode="markers", marker=dict(size=15, color=CORAL), name="Below 70%",
                                     hovertemplate="%{y}: %{x:.1f}%<extra>Below 70%</extra>"))
            fig.add_trace(go.Scatter(x=g["high"], y=names, mode="markers+text", marker=dict(size=15, color=TEAL), name="70% or more",
                                     text=[f"  +{v:.1f}" for v in g["gap"]], textposition="middle right",
                                     textfont=dict(color=NAVY, size=13), hovertemplate="%{y}: %{x:.1f}%<extra>70% or more</extra>"))
            fig.update_xaxes(title="Average score (% best answer)", range=[45, 97], ticksuffix="%")
            st.plotly_chart(style(fig, 380), width="stretch")
        with right:
            big = g["gap"].idxmax()
            rel = g["d"].idxmax()
            box("Takeaway", "Top hospitals are better across the whole stay, not in one area.", "dark")
            box("Biggest difference", f"<b style='font-size:22px;color:{CORAL}'>{g.loc[big, 'gap']:.1f} points</b><br>"
                f"Top hospitals score this much higher on {LABELS[big].lower()}.", "white")
            box("Most reliable difference", f"<b style='font-size:22px;color:{TEAL}'>{LABELS[rel]}</b><br>"
                f"Top hospitals almost always beat the rest here, even allowing for how much scores normally vary.", "white")
        with st.expander("See the numbers"):
            st.dataframe(g.rename(index=LABELS, columns={"high": "70% or more", "low": "Below 70%", "gap": "Points ahead",
                                                         "d": "Cohen’s d (size of gap)"}).iloc[::-1].round(2), width="stretch")

    with t3:
        st.markdown('<p class="question">Do recommendation rates differ by region and survey volume?</p>', unsafe_allow_html=True)
        a, b = st.columns(2, gap="medium")
        if "census_region" in df.columns:
            r = df.groupby("census_region")[TARGET].agg(["mean", "count"]).sort_values("mean")
            with a, st.container(border=True):
                chart_title("Region barely matters", "Average share of patients who would definitely recommend, by US region")
                fig = go.Figure(go.Bar(x=r["mean"], y=r.index, orientation="h",
                                       marker_color=[MIST if n < 30 else STEEL for n in r["count"]],
                                       text=[f"{v:.1f}%  ({n:,} hospitals)" for v, n in zip(r["mean"], r["count"])],
                                       textposition="outside"))
                fig.update_xaxes(range=[0, 95], ticksuffix="%")
                st.plotly_chart(style(fig, 300, legend=False), width="stretch")
                st.caption("Light bar: too few hospitals to draw conclusions.")
        if "completed_surveys" in df.columns:
            v = df.groupby(np.where(df["completed_surveys"] < 300, "Fewer than 300 surveys", "300 or more surveys"))[TARGET].agg(["mean", "count"])
            with b, st.container(border=True):
                chart_title("Hospitals with fewer surveys score a little higher", "Survey volume stands in for size: the data has no bed count")
                fig = go.Figure(go.Bar(x=v["mean"], y=v.index, orientation="h", marker_color=[TEAL, STEEL],
                                       text=[f"{m:.1f}%  ({n:,} hospitals)" for m, n in zip(v["mean"], v["count"])],
                                       textposition="outside"))
                fig.update_xaxes(range=[0, 95], ticksuffix="%")
                st.plotly_chart(style(fig, 300, legend=False), width="stretch")
        box(None, "Benchmark each hospital against similar hospitals, not only the national average.", "tint")

    with t4:
        st.markdown('<p class="question">How do the scores move together, and are there unusual hospitals?</p>', unsafe_allow_html=True)
        a, b = st.columns([1.3, 1], gap="medium")
        cols = PREDICTORS + [TARGET]
        corr = df[cols].corr()
        names = [SHORT[c] for c in PREDICTORS] + ["Recommend"]
        with a, st.container(border=True):
            chart_title("All six scores rise together with recommendation",
                        "1 = move perfectly together, 0 = unrelated. Darker = stronger link.")
            fig = go.Figure(go.Heatmap(z=corr.values, x=names, y=names, colorscale=[[0, "#FFFFFF"], [1, TEAL_DARK]],
                                       zmin=0, zmax=1, text=corr.round(2).values, texttemplate="%{text}", showscale=False))
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(style(fig, 420, legend=False), width="stretch")
        with b:
            best = corr[TARGET].drop(TARGET).idxmax()
            box("Strongest link", f"{LABELS[best]} (r = {corr.loc[best, TARGET]:.2f}). The communication scores also overlap "
                "heavily with each other, which matters when reading the models.", "white")
            q1, q3 = df[cols].quantile(.25), df[cols].quantile(.75)
            iqr = q3 - q1
            out = ((df[cols] < q1 - 1.5 * iqr) | (df[cols] > q3 + 1.5 * iqr)).sum()
            st.markdown('<div class="box white"><h4>Unusual hospitals (IQR rule)</h4>' +
                        bars_html([(LABELS.get(c, "Recommendation %"), out[c] / out.max() * 100, f"{out[c]}", STEEL) for c in cols])
                        + "<p>We kept them all: the values are real survey results, and the weakest hospitals are the ones leaders most need to see.</p></div>",
                        unsafe_allow_html=True)


def driver_table():
    # Rank the six scores with five methods and combine them
    coef = pd.Series(M["linear"].coef_, index=PREDICTORS)
    pooled = np.sqrt(((len(HI) - 1) * HI[PREDICTORS].var() + (len(LO) - 1) * LO[PREDICTORS].var()) / (len(HI) + len(LO) - 2))
    d = (HI[PREDICTORS].mean() - LO[PREDICTORS].mean()) / pooled
    odds = pd.Series(np.exp(M["logistic"].named_steps["logistic"].coef_[0]), index=PREDICTORS)
    measures = pd.DataFrame({
        "Moves with": df[PREDICTORS].corrwith(df[TARGET]),
        "Top vs rest": d,
        "Formula": coef,
        "Tree": pd.Series(M["tree_reg"].feature_importances_, index=PREDICTORS),
        "Odds of 70%+": odds,
    })
    ranks = measures.rank(ascending=False, method="first").astype(int)
    order = ranks.assign(avg=ranks.mean(axis=1)).sort_values(["avg", "Odds of 70%+"]).index
    return ranks.loc[order], coef, odds, d


def page_drivers():
    header("Key finding", "What drives patients to recommend a hospital",
           "The six parts of the stay, ranked by how strongly each is linked to patients recommending the hospital. "
           "The ranking comes from five different methods, which largely agree.")
    ranks, coef, odds, d = driver_table()
    sd = M["train_sd"]
    gap = HI[PREDICTORS].mean() - LO[PREDICTORS].mean()
    text = {
        "nurse_communication": (f"Biggest single link: each extra point goes with about {coef['nurse_communication']:.1f} of a point more recommendation.",
                                "coach nurses to listen and explain clearly"),
        "discharge_information": (f"Clearest sign a hospital clears 70%: about {sd['discharge_information']:.0f} points better multiplies the odds by {odds['discharge_information']:.1f}.",
                                  "plain-language take-home instructions"),
        "doctor_communication": (f"Close behind: top hospitals lead the rest by about {gap['doctor_communication']:.0f} points here.",
                                 "make time for questions at the bedside"),
        "night_quietness": (f"Lowest national score ({MEANS['night_quietness']:.0f}%) with the most room to grow, but a smaller link once communication is known.",
                            "quiet-hours plans on the wards"),
        "room_cleanliness": ("Helps, but less than communication does.", "keep cleaning standards visible"),
        "medicine_explanation": ("Adds nothing once nurse and doctor communication are known: hospitals that explain medicines well usually communicate well overall.",
                                 "fold it into communication training"),
    }
    left, right = st.columns([1.55, 1], gap="medium")
    with left:
        rows = ""
        for i, c in enumerate(ranks.index, 1):
            avg = ranks.loc[c].mean()
            if coef[c] < 0.05:
                tag, tag_style, dot = "No extra effect", "color:#7A2E14;background:#F6DED4", MIST
            elif avg <= 1.6:
                tag, tag_style, dot = "Strongest", f"color:{TEAL_DARK};background:#CDEBE6", TEAL
            elif avg <= 3:
                tag, tag_style, dot = "Strong", f"color:{TEAL_DARK};background:#CDEBE6", TEAL
            else:
                tag, tag_style, dot = "Moderate", "color:#1F4E74;background:#DCE7F1", STEEL
            sentence, action = text[c]
            rows += (f'<div class="driver"><span class="rank" style="background:{dot}">{i}</span><div class="body">'
                     f'<div class="top"><b>{LABELS[c]}</b><span class="tag" style="{tag_style}">{tag}</span></div>'
                     f'<p>{sentence}</p><p class="act">Action: {action}</p></div></div>')
        st.markdown(f'<div class="box white" style="padding:6px 22px">{rows}</div>', unsafe_allow_html=True)
    with right:
        cells = '<span></span>' + "".join(f'<span class="h">{m}</span>' for m in ranks.columns)
        for c in ranks.index:
            cells += f'<span style="font-weight:600">{SHORT[c]}</span>'
            for r in ranks.loc[c]:
                bg, fg = (TEAL_DARK, PAPER) if r == 1 else (("#7FC4BE", NAVY) if r <= 3 else (TINT, BODY))
                cells += f'<span class="r" style="background:{bg};color:{fg}">{r}</span>'
        st.markdown(f"""<div class="box white"><h4>Why we trust this ranking</h4>
<p style="font-size:12px;color:{MUTED};margin-bottom:10px">Each column is a different way of measuring the link. Numbers are the rank each method gave (1 = strongest).</p>
<div class="grid-rank">{cells}</div>
<p style="font-size:12px;color:{MUTED};margin-top:10px">Communication sits at the top in every column. Medicines only look strong on their own; once the other scores are counted, they add nothing.</p></div>""",
                    unsafe_allow_html=True)
        box("The bottom line", "Patients recommend hospitals that <b style='color:#F6F9FA'>communicate well</b>. "
            "Start with nurses and discharge instructions; quiet and cleanliness come next.", "dark")
        st.page_link(P_PREDICT, label="Test a change on your hospital", icon=":material/arrow_forward:")
    st.markdown('<p class="foot">These are links found in the data, not proof of cause. Pilot each change and measure the result.</p>',
                unsafe_allow_html=True)


def page_tiers():
    header("Peer groups", "Hospitals fall into three quality tiers",
           "We combined the six scores into an overall grade (PCA), then grouped look-alike hospitals (k-means). "
           "Recommendation was not used to form the groups.")
    X = df[PREDICTORS]
    pcs = M["pca"].transform(M["pca_scaler"].transform(X))
    sign = 1 if M["pca"].components_[0].sum() > 0 else -1
    tiers = pd.Series(tier_of(M, df), index=df.index)
    left, right = st.columns([1.5, 1], gap="medium")
    with left, st.container(border=True):
        chart_title("Each dot is a hospital; better overall experience to the right",
                    f"The left-to-right overall grade captures {M['pca_variance'][0]:.0%} of all differences between hospitals.")
        fig = go.Figure()
        names = hospital_labels(df)
        for t in TIER_ORDER:
            m = tiers == t
            fig.add_trace(go.Scattergl(x=pcs[m, 0] * sign, y=pcs[m, 1], mode="markers", name=f"{t} tier",
                                       marker=dict(size=6, color=TIER_COLORS[t], opacity=0.6),
                                       text=names[m], hovertemplate="%{text}<extra></extra>"))
        fig.update_xaxes(title="Overall patient experience →", showticklabels=False)
        fig.update_yaxes(title="", showticklabels=False, showgrid=False)
        st.plotly_chart(style(fig, 420), width="stretch")
    with right:
        st.markdown(f'<p class="chart-title" style="margin-bottom:8px">Patients who recommend, by tier</p>', unsafe_allow_html=True)
        for t in TIER_ORDER:
            sub = df[tiers == t]
            st.markdown(f'<div class="result" style="background:{CARD};box-shadow:0 4px 18px rgba(11,37,64,0.07);border-left:6px solid {TIER_COLORS[t]}">'
                        f'<div class="txt"><b style="color:{NAVY}">{t} tier</b><span style="color:{BODY}">{len(sub):,} hospitals</span></div>'
                        f'<div class="num"><span class="big" style="color:{TIER_COLORS[t]}">{sub[TARGET].mean():.0f}%</span></div></div>',
                        unsafe_allow_html=True)
        box(None, "Good hospitals tend to be good at everything. Each hospital can compare itself with its own tier.", "tint")
    with st.expander("Average scores in each tier"):
        prof = df.groupby(tiers)[PREDICTORS + [TARGET]].mean().reindex(TIER_ORDER).rename(columns={**LABELS, TARGET: "Recommendation %"})
        st.dataframe(prof.round(1), width="stretch")
    with st.expander("Why three groups?"):
        sil = pd.Series(M["silhouette"])
        fig = go.Figure(go.Scatter(x=sil.index, y=sil.values, mode="lines+markers", line=dict(color=TEAL, width=3)))
        fig.update_xaxes(title="Number of groups", dtick=1)
        fig.update_yaxes(title="How clearly groups separate")
        st.plotly_chart(style(fig, 280, legend=False), width="stretch")
        st.caption("Two groups separate most cleanly but only repeat the above/below-70% split. Three adds a useful middle tier. "
                   "All scores are moderate, so hospitals form a continuum rather than separate types.")


def page_predictor():
    header("Minimum viable product", "Hospital predictor",
           "Pick a hospital, change its scores, and see what our models expect to happen.", color=CORAL)
    labels = hospital_labels(df)
    options = ["National average hospital"] + sorted(labels.tolist())
    choice = st.selectbox("Start from a real hospital, or the national average", options)
    if choice == "National average hospital":
        start, actual = MEANS.round(0), None
    else:
        row = df[labels == choice].iloc[0]
        start, actual = row[PREDICTORS].astype(float), float(row[TARGET])
    if st.session_state.get("_sim_choice") != choice:
        for c in PREDICTORS:
            st.session_state[f"sim_{c}"] = int(round(start[c]))
        st.session_state["_sim_choice"] = choice

    left, right = st.columns([0.85, 1.25], gap="medium")
    with left, st.container(border=True):
        st.markdown(f'<div class="kicker" style="color:{MUTED}">Step 1</div><p class="chart-title">Set the hospital’s scores</p>'
                    f'<p class="chart-sub">% of patients giving the best answer on each part of the stay</p>', unsafe_allow_html=True)
        vals = {c: st.slider(LABELS[c], 0, 100, key=f"sim_{c}") for c in PREDICTORS}
        if st.button("Reset to starting point"):
            for c in PREDICTORS:
                st.session_state[f"sim_{c}"] = int(round(start[c]))
            st.rerun()

    X0 = pd.DataFrame([start.values], columns=PREDICTORS)
    X1 = pd.DataFrame([[vals[c] for c in PREDICTORS]], columns=PREDICTORS)
    p0, p1 = float(M["linear"].predict(X0)[0]), float(np.clip(M["linear"].predict(X1)[0], 0, 100))
    q0, q1 = float(M["logistic"].predict_proba(X0)[0, 1]), float(M["logistic"].predict_proba(X1)[0, 1])
    tier = tier_of(M, X1)[0]
    tier_avg = df[pd.Series(tier_of(M, df), index=df.index) == tier][TARGET].mean()
    changed = any(vals[c] != int(round(start[c])) for c in PREDICTORS)

    with right:
        st.markdown(f'<div class="kicker" style="color:{MUTED}">Step 2 · What our models expect</div>', unsafe_allow_html=True)
        before = f"was {p0:.1f}% before the change" if changed else (f"actual: {actual:.0f}%" if actual is not None else "national average hospital")
        st.markdown(f"""<div class="result" style="background:{NAVY}"><div class="txt"><b style="color:{PAPER}">Predicted recommendation rate</b>
<span style="color:#CFE0EA">Share of this hospital’s patients we expect would definitely recommend it</span></div>
<div class="num"><span class="big" style="color:{PAPER}">{p1:.1f}%</span><small style="color:#7FD3C9">{before}</small></div></div>""",
                    unsafe_allow_html=True)
        in10 = round(q1 * 10)
        was = f", was {q0:.0%}" if changed else ""
        st.markdown(f"""<div class="result" style="background:{CARD};box-shadow:0 4px 18px rgba(11,37,64,0.07);flex-direction:column;align-items:stretch;gap:10px">
<div style="display:flex;align-items:center;gap:20px"><div class="txt"><b style="color:{NAVY}">Chance of clearing the 70% line</b>
<span style="color:{BODY}">How likely the hospital is to land in the better-recommended half of US hospitals (70% or more)</span></div>
<div class="num"><span class="big" style="color:{TEAL}">{in10} in 10</span><small style="color:{TEAL}">{q1:.0%}{was}</small></div></div>
<div style="height:14px;border-radius:7px;background:{TINT};position:relative"><div style="width:{q1*100:.0f}%;height:14px;border-radius:7px;background:{TEAL}"></div>
{f'<div style="position:absolute;left:{q0*100:.0f}%;top:-4px;width:3px;height:22px;background:{NAVY}"></div>' if changed else ''}</div>
<div style="display:flex;justify-content:space-between;font-size:12px;color:{MUTED}"><span>Unlikely</span><span>{'Black mark: before the change' if changed else ''}</span><span>Very likely</span></div></div>""",
                    unsafe_allow_html=True)
        a, b = st.columns(2)
        with a:
            box(f"Peer group: <span style='color:{TIER_COLORS[tier]}'>{tier}</span>",
                f"Hospitals with similar scores. On average, {tier_avg:.0f}% of their patients recommend them.", "white")
        with b:
            if q1 < 0.5:
                box(None, "<b>At risk.</b> The model expects this hospital to fall below the 70% line.", "risk")
            elif q1 < 0.65:
                box(None, "<b>Too close to call.</b> The hospital sits around the 70% line, where predictions are least certain.", "warn")
            else:
                box(None, "<b>Likely to stay above 70%.</b> Keep protecting the strongest scores.", "good")

    gains = {}
    for c in PREDICTORS:
        Xg = X1.copy()
        Xg[c] = min(100, Xg[c].iloc[0] + 5)
        gains[c] = float(M["logistic"].predict_proba(Xg)[0, 1])
    g = pd.Series(gains).sort_values(ascending=False)
    rows = []
    for c, v in g.items():
        diff = (v - q1) * 100
        color = TEAL if diff >= 8 else (STEEL if diff >= 2 else (MIST if diff >= 0.5 else CORAL))
        rows.append((LABELS[c], v * 100, f"{v:.0%} ({'+' if diff >= 0.5 else ''}{diff:.0f})" if abs(diff) >= 0.5 else f"{v:.0%} (no gain)", color))
    with st.container(border=True):
        chart_title("If you could improve one area by 5 points, which helps most?",
                    f"Chance of clearing the 70% line after the improvement (today: {q1:.0%})")
        st.markdown(bars_html(rows), unsafe_allow_html=True)
    with st.expander("What the other models say"):
        verdict = "above" if M["class_tree"].predict(X1)[0] == 1 else "below"
        st.markdown(f"- **Classification tree:** {verdict} the 70% line\n"
                    f"- **Random forest:** {M['forest'].predict_proba(X1)[0, 1]:.0%} chance of clearing 70%\n"
                    f"- **Decision tree (number):** {M['tree_reg'].predict(X1)[0]:.1f}% predicted recommendation")
    st.markdown('<p class="foot">Estimates from links in the data, not guarantees. Pilot and measure any real change.</p>', unsafe_allow_html=True)


def metric_tables(reg, cls):
    # Technical metrics with plain meanings beside them
    html_out = ""
    if reg:
        names = list(reg)
        meaning = {"R²": "Share of hospital differences explained (1 = all)", "MAE": "Average miss, in percentage points",
                   "RMSE": "Typical miss, with big misses counted more"}
        cells = f'<span class="hd" style="color:{NAVY};font-weight:700">Predicting the %</span>' + "".join(f'<span class="hd">{n.split()[0]}</span>' for n in names) + '<span class="hd">What it means</span>'
        for k, lower in [("R²", False), ("MAE", True), ("RMSE", True)]:
            vals = [reg[n][k] for n in names]
            best = int(np.argmin(vals) if lower else np.argmax(vals))
            cells += f'<span style="font-weight:600">{k}</span>' + "".join(
                f'<span class="{"best" if i == best else ""}">{v:.3f}</span>' if k == "R²" else f'<span class="{"best" if i == best else ""}">{v:.2f}</span>'
                for i, v in enumerate(vals)) + f'<span class="mean">{meaning[k]}</span>'
        html_out += f'<div class="box white"><div class="mtab" style="grid-template-columns:170px repeat({len(names)},90px) 1fr">{cells}</div></div>'
    if cls:
        names = [n for n in cls if not n.startswith("_")]
        meaning = {"Accuracy": "Share of hospitals called right", "Precision": "When it says “70%+”, how often it is right",
                   "Recall": "Of the real 70%+ hospitals, the share it found", "F1": "One score balancing precision and recall",
                   "ROC-AUC": "Ranking skill: 0.5 is a coin flip, 1 is perfect"}
        short = {"Logistic regression": "Logistic", "Classification tree": "Tree", "Random forest": "Forest"}
        cells = f'<span class="hd" style="color:{NAVY};font-weight:700">Flagging below 70%</span>' + "".join(f'<span class="hd">{short[n]}</span>' for n in names) + '<span class="hd">What it means</span>'
        for k in meaning:
            vals = [cls[n][k] for n in names]
            best = int(np.nanargmax(vals))
            cells += f'<span style="font-weight:600">{k}</span>' + "".join(
                f'<span class="{"best" if i == best else ""}">{v:.3f}</span>' for i, v in enumerate(vals)) + f'<span class="mean">{meaning[k]}</span>'
        html_out += f'<div class="box white"><div class="mtab" style="grid-template-columns:170px repeat({len(names)},80px) 1fr">{cells}</div></div>'
    st.markdown(html_out, unsafe_allow_html=True)


def page_test():
    header("Model test · for technical reviewers", "Upload hospitals, let the trained models run")
    left, right = st.columns([0.8, 1.9], gap="medium")
    with left:
        st.markdown(f"""<div class="box white"><div class="kicker" style="color:{MUTED}">1 · Saved models</div>
<p><b style="color:{TEAL_DARK}">✓ 5 models loaded, no retraining</b><br>Trained once on {M['n_train']:,} hospitals (60%, seed {M['seed']}) and {('loaded from the saved file' if model_source == 'saved' else 'trained when the app started')}.</p></div>""",
                    unsafe_allow_html=True)
        st.markdown(f'<div class="kicker" style="color:{MUTED};margin-top:6px">2 · Choose hospitals</div>', unsafe_allow_html=True)
        source = st.radio("Hospitals to test", [f"Our held-out test hospitals ({len(cls_test):,})", "Upload a CSV"],
                          label_visibility="collapsed")
        frame, note = None, ""
        if source.startswith("Upload"):
            up = st.file_uploader("CSV with the six score columns", type="csv")
            if up is not None:
                frame = pd.read_csv(up, dtype={"facility_id": str})
                missing = [c for c in PREDICTORS if c not in frame.columns]
                if missing:
                    st.error("Missing columns: " + ", ".join(missing))
                    frame = None
                else:
                    frame = frame.dropna(subset=PREDICTORS)
                    note = f"{len(frame):,} hospitals · 6 scores found"
            with st.expander("Columns the file needs"):
                st.code(", ".join(PREDICTORS) + f"\n\nOptional: {TARGET} (actual %), facility_name, city, state")
        else:
            frame, note = cls_test, f"{len(cls_test):,} hospitals the models never saw"
        if note:
            st.caption(note)
        run = st.button("Run the test", type="primary", width="stretch", disabled=frame is None)
        st.caption("No actual results in the file? The page still scores every hospital and flags risk; it just skips the grading.")
    if run:
        st.session_state["test_frame"] = frame
        st.session_state["test_source"] = source
    with right:
        frame = st.session_state.get("test_frame")
        if frame is None or st.session_state.get("test_source") != source:
            box("Results appear here", "Choose the hospitals on the left, then press <b>Run the test</b>.", "tint")
            return
        graded = TARGET in frame.columns
        if graded:
            if source.startswith("Our"):
                reg, cls, _ = REG, CLS, CMS
            else:
                reg, cls, _ = evaluate(M, frame, frame)
                if "facility_id" in frame.columns:
                    trained = set(id_col) - set(M["cls_test_ids"])
                    overlap = frame["facility_id"].isin(trained).sum()
                    if overlap:
                        st.warning(f"{overlap:,} of these hospitals were used to train the models, so these scores will look better than reality.")
            metric_tables(reg, cls)
        else:
            box(None, "No actual recommendation column found, so the models score the hospitals without grading.", "warn")
        scored = score_hospitals(M, frame)
        show = pd.DataFrame({"Hospital": hospital_labels(scored)})
        if graded:
            show["Actual %"] = scored[TARGET].values
        show["Predicted %"] = scored["predicted_recommend_pct"].values
        show["Chance 70%+"] = (scored["chance_70_or_more"] * 100).round(0).astype(int).astype(str) + "%"
        show["Peer tier"] = scored["peer_tier"].values
        show["Flag"] = scored["flag"].values
        with st.container(border=True):
            top = st.columns([2, 1])
            top[0].markdown(f'<p class="chart-title">Scored hospitals ({len(show):,})</p>', unsafe_allow_html=True)
            top[1].download_button("Download scored CSV", scored.to_csv(index=False), "scored_hospitals.csv", "text/csv", width="stretch")
            st.dataframe(show, hide_index=True, width="stretch", height=300)


def page_performance():
    header("Model scores", "How well our models scored on hospitals they had never seen",
           f"Each model learned from 60% of hospitals ({M['n_train']:,}). We then asked it about the other 40% "
           f"({M['n_test']:,}) and compared its answers with what really happened.", color=NAVY)
    a, b = st.columns(2, gap="medium")
    lin, tree = REG["Linear regression"], REG["Decision tree"]
    with a:
        st.markdown(f"""<div class="box white"><div class="kicker">Task 1 · Predict the recommendation rate</div>
<h4>How far off were the predictions, on average?</h4>""" + bars_html([
            ("Linear regression", lin["MAE"] * 10, f"{lin['MAE']:.1f} points off", TEAL),
            ("Decision tree", tree["MAE"] * 10, f"{tree['MAE']:.1f} points off", MIST)]) +
            f"""<p class="foot" style="margin:0 0 10px">Shorter bar is better</p>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
<div class="box tint" style="margin:0;padding:12px 14px"><b style="font-size:20px;color:{NAVY}">{round(lin['within10']*10)} in 10</b><p style="font-size:12px">predictions are within 10 points of the real rate. A hospital really at 70% is usually predicted between 60% and 80%.</p></div>
<div class="box tint" style="margin:0;padding:12px 14px"><b style="font-size:20px;color:{NAVY}">{lin['R²']:.0%}</b><p style="font-size:12px">of why some hospitals are recommended more than others is explained by the six scores. The rest comes from things the survey does not measure, like staffing.</p></div>
</div></div>""", unsafe_allow_html=True)
    with b:
        rows = [(n, CLS[n]["Accuracy"] * 100, f"{CLS[n]['Accuracy']*100:.0f} in 100", TEAL if i == 0 else MIST)
                for i, n in enumerate(["Logistic regression", "Classification tree", "Random forest"])]
        st.markdown(f"""<div class="box white"><div class="kicker">Task 2 · Flag hospitals below 70%</div>
<h4>Out of 100 hospitals, how many were put on the right side of the line?</h4>""" + bars_html(rows) +
                    f"""<p class="foot">Longer bar is better. Blind guessing would get {CLS['_baseline']*100:.0f} in 100.</p></div>""",
                    unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1.35, 1, 1], gap="medium")
    tn, fp, fn, tp = CMS["Logistic regression"].ravel()
    with c1:
        st.markdown(f"""<div class="box white"><h4>What the {tn+fp+fn+tp:,} calls looked like</h4>
<p class="chart-sub" style="margin-bottom:10px">Logistic regression, our recommended flagging model</p>
<div class="cm">
<div style="background:{TEAL};color:{PAPER}"><span class="n">{tp}</span><b>Correctly reassured</b><small style="color:#CDEBE6">Said 70%+, and it was</small></div>
<div style="background:{TEAL};color:{PAPER}"><span class="n">{tn}</span><b>Correctly warned</b><small style="color:#CDEBE6">Said below 70%, and it was</small></div>
<div style="background:#F6DED4;color:#7A2E14"><span class="n">{fp}</span><b>Missed warning</b><small>Said 70%+, but it fell below: the costly mistake</small></div>
<div style="background:{TINT};color:{INK}"><span class="n">{fn}</span><b>Unneeded warning</b><small style="color:{BODY}">Said below 70%, but it was fine</small></div>
</div><p class="foot">Most mistakes are hospitals sitting right next to the 70% line.</p></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="box tint"><h4>Which model we recommend for each job</h4>
<p><b style="color:{TEAL}">To predict a hospital’s recommendation %: linear regression</b><br>Closest predictions, and a simple formula anyone can read.</p><br>
<p><b style="color:{TEAL}">To warn which hospitals may fall below 70%: logistic regression</b><br>Within about 1 point of the random forest, and it can explain why a hospital was flagged.</p>
<p class="foot">Technical scores (R², RMSE, precision, recall, ROC-AUC) are on the Test the models page.</p></div>""",
                    unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="box dark"><h4>Use with care</h4>
<p><b style="color:{PAPER}">Links, not proof.</b> Raising a score may not lift recommendation by the full amount.</p><br>
<p><b style="color:{PAPER}">One year of averages.</b> Describes hospitals, not individual patients.</p><br>
<p><b style="color:{PAPER}">Missing factors.</b> No size, staffing or patient mix in the data.</p></div>""", unsafe_allow_html=True)


def page_methods():
    header("Methods and limits", "How we did it, and what it cannot tell you", color=NAVY)
    a, b = st.columns(2, gap="medium")
    with a:
        st.markdown(f"""<div class="box white"><h4>Data</h4><p>CMS HCAHPS Hospital survey (public, data.cms.gov), patients discharged
October 2024 to September 2025. Reshaped to one row per hospital; kept hospitals with 100+ completed surveys and no CMS
data-collection flags: {N:,} hospitals.</p></div>
<div class="box white"><h4>Fair testing</h4><p>Every model learned from 60% of hospitals and was scored on the 40% it never saw
(random seed {M['seed']}; stratified for the yes/no models). Settings were tuned with 5-fold cross-validation on the learning set only.</p></div>
<div class="box white"><h4>Models</h4><p><b>Peer tiers:</b> PCA + k-means (3 groups). <b>Recommendation %:</b> linear regression; decision tree
(depth {M['tree_reg_params']['max_depth']}). <b>Below 70%:</b> logistic regression (C = {M['logistic_params']['logistic__C']}); classification tree
(depth {M['class_tree_params']['max_depth']}); random forest (300 trees).</p></div>""", unsafe_allow_html=True)
    with b:
        st.markdown("""<div class="box dark"><h4>Limitations</h4>
<p>• <b style="color:#F6F9FA">Association, not causation:</b> one year of observational data.</p>
<p>• <b style="color:#F6F9FA">Hospital averages:</b> results describe hospitals, not individual patients.</p>
<p>• <b style="color:#F6F9FA">Excluded hospitals:</b> those with fewer than 100 surveys.</p>
<p>• <b style="color:#F6F9FA">Missing factors:</b> no size, staffing, case mix; two survey composites were dropped by CMS.</p>
<p>• <b style="color:#F6F9FA">Overlapping scores:</b> the communication measures move together, so they share credit.</p>
<p>• <b style="color:#F6F9FA">The 70% line</b> is the national middle, not a clinical standard.</p></div>
<div class="box tint"><h4>How AI helped</h4><p>We used AI assistants (ChatGPT and Claude) for coding, debugging, merging our notebooks and
plain-language explanations. Every number was produced by running our code on the real data and checked by the group.</p></div>""",
                    unsafe_allow_html=True)


# ───────────────────────────────────────────── navigation
P_OVERVIEW = st.Page(page_overview, title="Overview", icon=":material/home:", url_path="overview", default=True)
P_EXPLORE = st.Page(page_explore, title="Explore the data", icon=":material/bar_chart:", url_path="explore")
P_DRIVERS = st.Page(page_drivers, title="What drives recommendation", icon=":material/insights:", url_path="drivers")
P_TIERS = st.Page(page_tiers, title="Peer tiers", icon=":material/groups:", url_path="tiers")
P_PREDICT = st.Page(page_predictor, title="Hospital predictor", icon=":material/tune:", url_path="predictor")
P_TEST = st.Page(page_test, title="Test the models", icon=":material/science:", url_path="test")
P_PERF = st.Page(page_performance, title="How the models performed", icon=":material/fact_check:", url_path="performance")
P_METHODS = st.Page(page_methods, title="Methods and limits", icon=":material/menu_book:", url_path="methods")

nav = st.navigation({"Story": [P_OVERVIEW, P_EXPLORE, P_DRIVERS, P_TIERS],
                     "Tool": [P_PREDICT, P_TEST, P_PERF, P_METHODS]})
st.sidebar.markdown(f"""<div style="margin-top:24px;font-size:12px;line-height:1.6;color:{MIST}">
<b style="font-family:'DM Serif Display',Georgia,serif;font-size:18px;color:#F6F9FA;font-weight:400">Recommend</b><br>
HCAHPS analytics<br><br>Rebecca Quaye · Ketty Odero · Victoria Kutuusa<br>CMS HCAHPS · Oct 2024 to Sep 2025</div>""",
                    unsafe_allow_html=True)
nav.run()
