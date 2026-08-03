
from __future__ import annotations

from pathlib import Path
import csv
import io

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine import BasalHeaveEngine, DatabaseError, load_csv_database


APP_DIR = Path(__file__).resolve().parent
DATABASE_PATH = APP_DIR / "data" / "basal_heave_database_4200.csv"


@st.cache_resource(show_spinner=False)
def get_engine() -> BasalHeaveEngine:
    return BasalHeaveEngine(load_csv_database(DATABASE_PATH))


def basis_label(value: str) -> str:
    return {
        "lower": "Lower bound — conservative",
        "average": "Average of lower and upper bounds",
        "upper": "Upper bound",
    }[value]


st.set_page_config(
    page_title="Basal Heave Stability Calculator",
    page_icon="🏗️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1500px; padding-top: 1.4rem; padding-bottom: 3rem;}
    [data-testid="stMetric"] {
        background: #f7f9fc;
        border: 1px solid #dce4ec;
        border-radius: 12px;
        padding: 14px 16px;
    }
    .small-note {
        color: #536575;
        font-size: 0.92rem;
    }
    .scope-box {
        background: #f5f8fb;
        border-left: 5px solid #245b84;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

engine = get_engine()

st.title("Basal Heave Stability Calculator")
st.caption(
    "New 4,200-model database • Terzaghi • Bjerrum & Eide • "
    "Optum lower and upper bounds"
)

with st.expander("Database scope and interpolation domain", expanded=False):
    st.markdown(
        """
        <div class="scope-box">
        <b>Numerical scope:</b>
        Su = 5–60 kPa; γ = 16–20 kN/m³; q = 0–20 kPa;
        H = 6–14 m; B = 4–30 m; R<sub>int</sub> = 1.0.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write(
        "The only absent database corner is **q = 20 kPa with Su = 5 kPa**. "
        "For 10 < q ≤ 20 kPa, the supported lower boundary is **Su ≥ q/2**. "
        "The app uses triangular interpolation along that boundary and does not extrapolate."
    )

st.subheader("Input parameters")

input_cols = st.columns(5)
with input_cols[2]:
    q = st.number_input(
        "Uniform surcharge, q (kPa)",
        min_value=0.0, max_value=20.0, value=10.0, step=1.0,
    )

minimum_su = engine.minimum_su_for_q(q)
with input_cols[0]:
    Su = st.number_input(
        "Undrained shear strength, Su (kPa)",
        min_value=float(minimum_su), max_value=60.0,
        value=max(20.0, float(minimum_su)), step=1.0,
        help=f"For q={q:g} kPa, the minimum data-supported Su is {minimum_su:g} kPa.",
    )
with input_cols[1]:
    gamma = st.number_input(
        "Total unit weight, γ (kN/m³)",
        min_value=16.0, max_value=20.0, value=18.0, step=0.5,
    )
with input_cols[3]:
    H = st.number_input(
        "Excavation depth, H (m)",
        min_value=6.0, max_value=14.0, value=10.0, step=0.5,
    )
with input_cols[4]:
    B = st.number_input(
        "Excavation width, B (m)",
        min_value=4.0, max_value=30.0, value=10.0, step=0.5,
    )

settings_cols = st.columns([1.4, 1.0, 2.6])
with settings_cols[0]:
    recommendation_basis = st.selectbox(
        "Recommended Optum value",
        options=["lower", "average", "upper"],
        format_func=basis_label,
        index=1,
    )
with settings_cols[1]:
    target_fs = st.number_input(
        "Target F.S.",
        min_value=0.0, value=1.50, step=0.05,
    )

try:
    result = engine.calculate(
        Su=Su, gamma=gamma, q=q, H=H, B=B,
        recommendation_basis=recommendation_basis,
    )
except DatabaseError as exc:
    st.error(str(exc))
    st.stop()

st.divider()
st.subheader("Calculated factors of safety")

top = st.columns(3)
top[0].metric("Terzaghi F.S. — resistance form", f"{result['terzaghi_resistance']:.3f}")
top[1].metric("Bjerrum & Eide F.S.", f"{result['bjerrum']:.3f}")
top[2].metric("Recommended Optum F.S.", f"{result['recommended']:.3f}")

bounds = st.columns(4)
bounds[0].metric("Optum lower bound", f"{result['numerical_lower']:.3f}")
bounds[1].metric("Optum average", f"{result['numerical_average']:.3f}")
bounds[2].metric("Optum upper bound", f"{result['numerical_upper']:.3f}")
bounds[3].metric("Upper–lower gap", f"{result['bound_gap_percent']:.1f}%")

difference = result["recommended"] - target_fs
if difference >= 0:
    st.success(
        f"The recommended F.S. is {difference:.3f} above the user-defined target "
        f"of {target_fs:.2f}."
    )
else:
    st.error(
        f"The recommended F.S. is {abs(difference):.3f} below the user-defined target "
        f"of {target_fs:.2f}."
    )

details = pd.DataFrame([
    ["H/B", result["H_over_B"], "—"],
    ["B/H", result["B_over_H"], "—"],
    ["γH + q", result["driving_pressure"], "kPa"],
    ["Bjerrum Nc", result["Nc"], "—"],
    ["Recommendation basis", basis_label(recommendation_basis), "—"],
], columns=["Quantity", "Value", "Unit"])
st.dataframe(details, use_container_width=True, hide_index=True)

with st.expander("Second Terzaghi expression stored in the workbook"):
    if result["terzaghi_net_pressure"] is None:
        st.warning(
            "The original net-pressure denominator is zero or negative for this input, "
            "so that expression is not reported as a usable positive factor of safety."
        )
    else:
        st.metric(
            "Terzaghi F.S. — original net-pressure form",
            f"{result['terzaghi_net_pressure']:.3f}",
        )
    st.write(
        "The main Terzaghi result above uses the workbook formula "
        "(5.7Su + SuH/(B/√2))/(γH + q)."
    )

st.subheader("Family-of-curves comparison")

curve_rows = []
points = 121
for i in range(points):
    curve_B = 30.0 - (30.0 - 4.0) * i / (points - 1)
    curve = engine.calculate(
        Su=Su, gamma=gamma, q=q, H=H, B=curve_B,
        recommendation_basis=recommendation_basis,
    )
    curve_rows.append(curve)

curve_df = pd.DataFrame(curve_rows).sort_values("H_over_B")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=curve_df["H_over_B"], y=curve_df["terzaghi_resistance"],
    name="Terzaghi", mode="lines",
))
fig.add_trace(go.Scatter(
    x=curve_df["H_over_B"], y=curve_df["bjerrum"],
    name="Bjerrum & Eide", mode="lines",
))
fig.add_trace(go.Scatter(
    x=curve_df["H_over_B"], y=curve_df["numerical_lower"],
    name="Optum lower", mode="lines",
))
fig.add_trace(go.Scatter(
    x=curve_df["H_over_B"], y=curve_df["numerical_average"],
    name="Optum average", mode="lines",
))
fig.add_trace(go.Scatter(
    x=curve_df["H_over_B"], y=curve_df["numerical_upper"],
    name="Optum upper", mode="lines",
))
fig.add_vline(
    x=result["H_over_B"], line_dash="dash", line_color="firebrick",
    annotation_text="Selected H/B",
)
fig.update_layout(
    xaxis_title="H/B",
    yaxis_title="Factor of Safety",
    legend_title="Method",
    hovermode="x unified",
    height=520,
    margin=dict(l=20, r=20, t=30, b=20),
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Nearest modeled cases")
nearest = pd.DataFrame(engine.nearest_cases(
    Su=Su, gamma=gamma, q=q, H=H, B=B, count=8
))
st.dataframe(nearest, use_container_width=True, hide_index=True)

export_rows = [
    ("Su (kPa)", result["Su"]),
    ("gamma (kN/m3)", result["gamma"]),
    ("q (kPa)", result["q"]),
    ("H (m)", result["H"]),
    ("B (m)", result["B"]),
    ("H/B", result["H_over_B"]),
    ("B/H", result["B_over_H"]),
    ("Terzaghi resistance-form F.S.", result["terzaghi_resistance"]),
    ("Terzaghi net-pressure F.S.", result["terzaghi_net_pressure"]),
    ("Bjerrum Nc", result["Nc"]),
    ("Bjerrum & Eide F.S.", result["bjerrum"]),
    ("Optum lower bound", result["numerical_lower"]),
    ("Optum average", result["numerical_average"]),
    ("Optum upper bound", result["numerical_upper"]),
    ("Recommended Optum F.S.", result["recommended"]),
    ("Recommendation basis", basis_label(recommendation_basis)),
    ("Bound gap (%)", result["bound_gap_percent"]),
    ("Target F.S.", target_fs),
]
buffer = io.StringIO()
writer = csv.writer(buffer)
writer.writerow(["Item", "Value"])
writer.writerows(export_rows)
st.download_button(
    "Download calculation as CSV",
    data=buffer.getvalue().encode("utf-8"),
    file_name="basal_heave_calculation.csv",
    mime="text/csv",
    use_container_width=True,
)

st.divider()
st.subheader("Applicability")
st.markdown(
    """
- Homogeneous clay and constant undrained shear strength.
- Undrained total-stress condition.
- R_int = 1.0, matching the numerical database.
- The wall, support, boundary, groundwater, and hard-stratum assumptions must match the Optum models.
- Numerical results are interpolated only inside the data-supported domain; extrapolation is blocked.
- The target F.S. is user-defined and is not presented as a universal code requirement.
"""
)
