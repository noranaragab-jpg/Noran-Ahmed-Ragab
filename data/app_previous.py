from __future__ import annotations

from pathlib import Path
import csv
import io
import math
import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine import BasalHeaveEngine, DatabaseError, load_csv_database


APP_DIR = Path(__file__).resolve().parent
DATABASE_PATH = APP_DIR / "data" / "basal_heave_database_4200.csv"


@st.cache_resource(show_spinner=False)
def get_engine() -> BasalHeaveEngine:
    return BasalHeaveEngine(load_csv_database(DATABASE_PATH))


def normalize_result(result: dict, B: float) -> dict:
    """Make the app compatible with both the previous and revised engine files."""
    normalized = dict(result)
    normalized.setdefault("B1", B / math.sqrt(2.0))

    # Revised engine key. The previous engine stored the same requested
    # Terzaghi equation under ``terzaghi_net_pressure``.
    if "terzaghi" not in normalized:
        normalized["terzaghi"] = normalized.get("terzaghi_net_pressure")

    if "terzaghi_denominator" not in normalized:
        normalized["terzaghi_denominator"] = normalized.get(
            "terzaghi_net_pressure_denominator"
        )

    return normalized


def basis_label(value: str) -> str:
    return {
        "lower": "Lower bound — conservative",
        "average": "Average of lower and upper bounds",
        "upper": "Upper bound",
    }[value]


def format_fs(value: float | None) -> str:
    return "Not valid" if value is None else f"{value:.3f}"


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
    .scope-box {
        background: #f5f8fb;
        border-left: 5px solid #245b84;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0 1rem 0;
    }
    .equation-box {
        background: #fbfcfd;
        border: 1px solid #dce4ec;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

engine = get_engine()

st.title("Basal Heave Stability Calculator")
st.caption(
    "Terzaghi and Bjerrum & Eide calculated directly from equations • "
    "Only Optum results are interpolated from the 4,200-model database"
)

with st.expander("Calculation method and database scope", expanded=False):
    st.markdown(
        """
        <div class="scope-box">
        <b>Optum interpolation scope:</b>
        Su = 5–60 kPa; γ = 16–20 kN/m³; q = 0–20 kPa;
        H = 6–14 m; B = 4–30 m; R<sub>int</sub> = 1.0.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write(
        "The database is used only to interpolate the Optum lower and upper bounds. "
        "Terzaghi and Bjerrum & Eide are recalculated analytically for every input."
    )
    st.write(
        "The only absent Optum corner is q = 20 kPa with Su = 5 kPa. "
        "For 10 < q ≤ 20 kPa, the supported boundary is Su ≥ q/2."
    )

st.subheader("Input parameters")

input_cols = st.columns(5)
with input_cols[2]:
    q = st.number_input(
        "Uniform surcharge, q (kPa)",
        min_value=0.0,
        max_value=20.0,
        value=10.0,
        step=1.0,
    )

minimum_su = engine.minimum_su_for_q(q)
with input_cols[0]:
    Su = st.number_input(
        "Undrained shear strength, Su (kPa)",
        min_value=float(minimum_su),
        max_value=60.0,
        value=max(20.0, float(minimum_su)),
        step=1.0,
        help=(
            f"The Optum database requires Su ≥ {minimum_su:g} kPa "
            f"when q = {q:g} kPa."
        ),
    )
with input_cols[1]:
    gamma = st.number_input(
        "Total unit weight, γ (kN/m³)",
        min_value=16.0,
        max_value=20.0,
        value=18.0,
        step=0.5,
    )
with input_cols[3]:
    H = st.number_input(
        "Excavation depth, H (m)",
        min_value=6.0,
        max_value=14.0,
        value=10.0,
        step=0.5,
    )
with input_cols[4]:
    B = st.number_input(
        "Excavation width, B (m)",
        min_value=4.0,
        max_value=30.0,
        value=10.0,
        step=0.5,
    )

settings_cols = st.columns([1.5, 1.0, 2.5])
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
        min_value=0.0,
        value=1.50,
        step=0.05,
    )

try:
    result = normalize_result(
        engine.calculate(
            Su=Su,
            gamma=gamma,
            q=q,
            H=H,
            B=B,
            recommendation_basis=recommendation_basis,
        ),
        B,
    )
except DatabaseError as exc:
    st.error(str(exc))
    st.stop()

st.divider()
st.subheader("Calculated factors of safety")

main_results = st.columns(3)
main_results[0].metric("Terzaghi F.S. — equation", format_fs(result["terzaghi"]))
main_results[1].metric("Bjerrum & Eide F.S. — equation", f"{result['bjerrum']:.3f}")
main_results[2].metric("Recommended Optum F.S. — interpolation", f"{result['recommended']:.3f}")

if result["terzaghi"] is None:
    st.warning(
        "The Terzaghi denominator is zero or negative for this input: "
        f"γH + q − SuH/B₁ = {result['terzaghi_denominator']:.3f} kPa. "
        "A positive Terzaghi factor of safety is therefore not reported."
    )

optum_results = st.columns(4)
optum_results[0].metric("Optum lower bound", f"{result['numerical_lower']:.3f}")
optum_results[1].metric("Optum average", f"{result['numerical_average']:.3f}")
optum_results[2].metric("Optum upper bound", f"{result['numerical_upper']:.3f}")
optum_results[3].metric("Upper–lower gap", f"{result['bound_gap_percent']:.1f}%")

difference = result["recommended"] - target_fs
if difference >= 0:
    st.success(
        f"The recommended Optum F.S. is {difference:.3f} above the target "
        f"of {target_fs:.2f}."
    )
else:
    st.error(
        f"The recommended Optum F.S. is {abs(difference):.3f} below the target "
        f"of {target_fs:.2f}."
    )

st.subheader("Equations used")

equation_cols = st.columns(2)
with equation_cols[0]:
    st.markdown("**Terzaghi (1943)**")
    st.latex(r"B_1=\frac{B}{\sqrt{2}}")
    st.latex(
        r"F.S._T=\frac{5.7S_u}{\gamma H+q-\dfrac{S_uH}{B_1}}"
    )
    st.caption(
        "Homogeneous clay is assumed, so Cᵤb = Cᵤh = Su. "
        "No nearby hard stratum is included, so B₁ = B/√2."
    )

with equation_cols[1]:
    st.markdown("**Bjerrum & Eide (1956)**")
    st.latex(r"F.S._{B\&E}=\frac{S_uN_c}{\gamma H+q}")
    st.latex(
        r"N_c=\begin{cases}"
        r"5.14+1.05\left(\dfrac{H}{B}\right)-0.18\left(\dfrac{H}{B}\right)^2," 
        r"& \dfrac{H}{B}\leq3 \\"
        r"7.2,& \dfrac{H}{B}>3"
        r"\end{cases}"
    )

calculation_details = pd.DataFrame(
    [
        ["H/B", result["H_over_B"], "—"],
        ["B/H", result["B_over_H"], "—"],
        ["B₁ = B/√2", result["B1"], "m"],
        ["γH + q", result["driving_pressure"], "kPa"],
        ["Terzaghi denominator", result["terzaghi_denominator"], "kPa"],
        ["Bjerrum Nc", result["Nc"], "—"],
        ["Optum recommendation basis", basis_label(recommendation_basis), "—"],
    ],
    columns=["Quantity", "Value", "Unit"],
)
st.dataframe(calculation_details, use_container_width=True, hide_index=True)

st.subheader("Family-of-curves comparison")

curve_rows: list[dict[str, float | str | None]] = []
points = 121
for i in range(points):
    curve_B = 30.0 - (30.0 - 4.0) * i / (points - 1)
    curve_rows.append(
        normalize_result(
            engine.calculate(
                Su=Su,
                gamma=gamma,
                q=q,
                H=H,
                B=curve_B,
                recommendation_basis=recommendation_basis,
            ),
            curve_B,
        )
    )

curve_df = pd.DataFrame(curve_rows).sort_values("H_over_B")
# Plotly interprets NaN as a gap. This avoids plotting invalid negative-denominator Terzaghi values.
curve_df["terzaghi_plot"] = pd.to_numeric(curve_df["terzaghi"], errors="coerce")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=curve_df["H_over_B"],
    y=curve_df["terzaghi_plot"],
    name="Terzaghi — equation",
    mode="lines",
))
fig.add_trace(go.Scatter(
    x=curve_df["H_over_B"],
    y=curve_df["bjerrum"],
    name="Bjerrum & Eide — equation",
    mode="lines",
))
fig.add_trace(go.Scatter(
    x=curve_df["H_over_B"],
    y=curve_df["numerical_lower"],
    name="Optum lower — interpolation",
    mode="lines",
))
fig.add_trace(go.Scatter(
    x=curve_df["H_over_B"],
    y=curve_df["numerical_average"],
    name="Optum average — interpolation",
    mode="lines",
))
fig.add_trace(go.Scatter(
    x=curve_df["H_over_B"],
    y=curve_df["numerical_upper"],
    name="Optum upper — interpolation",
    mode="lines",
))
fig.add_vline(
    x=result["H_over_B"],
    line_dash="dash",
    line_color="firebrick",
    annotation_text="Selected H/B",
)
fig.update_layout(
    xaxis_title="H/B",
    yaxis_title="Factor of Safety",
    legend_title="Method",
    hovermode="x unified",
    height=540,
    margin=dict(l=20, r=20, t=30, b=20),
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Nearest original Optum models")
nearest = pd.DataFrame(
    engine.nearest_cases(Su=Su, gamma=gamma, q=q, H=H, B=B, count=8)
)
st.dataframe(nearest, use_container_width=True, hide_index=True)

export_rows = [
    ("Su (kPa)", result["Su"]),
    ("gamma (kN/m3)", result["gamma"]),
    ("q (kPa)", result["q"]),
    ("H (m)", result["H"]),
    ("B (m)", result["B"]),
    ("H/B", result["H_over_B"]),
    ("B/H", result["B_over_H"]),
    ("B1=B/sqrt(2) (m)", result["B1"]),
    ("Terzaghi F.S. calculated from equation", result["terzaghi"]),
    ("Terzaghi denominator (kPa)", result["terzaghi_denominator"]),
    ("Bjerrum Nc calculated from equation", result["Nc"]),
    ("Bjerrum & Eide F.S. calculated from equation", result["bjerrum"]),
    ("Optum lower bound interpolated", result["numerical_lower"]),
    ("Optum average interpolated", result["numerical_average"]),
    ("Optum upper bound interpolated", result["numerical_upper"]),
    ("Recommended Optum F.S.", result["recommended"]),
    ("Recommendation basis", basis_label(recommendation_basis)),
    ("Upper-lower gap (%)", result["bound_gap_percent"]),
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
- Homogeneous clay with constant undrained shear strength, so Cᵤb = Cᵤh = Su.
- Undrained total-stress condition.
- No nearby hard stratum is included; therefore B₁ = B/√2 in the Terzaghi equation.
- Surcharge q is added to the driving pressure in both classical equations.
- R_int = 1.0, matching the Optum numerical database.
- Only Optum lower and upper bounds are interpolated from the database.
- Terzaghi and Bjerrum & Eide are calculated directly from equations for every input.
- Optum extrapolation outside the database domain is blocked.
"""
)
