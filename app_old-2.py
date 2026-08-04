from __future__ import annotations

from pathlib import Path
import csv
import io
import math

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from engine import BasalHeaveEngine, DatabaseError, load_csv_database


APP_DIR = Path(__file__).resolve().parent
DATABASE_PATH = APP_DIR / "data" / "basal_heave_database_4200.csv"


@st.cache_resource(show_spinner=False)
def get_engine() -> BasalHeaveEngine:
    return BasalHeaveEngine(load_csv_database(DATABASE_PATH))


def normalize_result(result: dict, B: float) -> dict:
    """Keep compatibility with both the previous and revised engine versions."""
    normalized = dict(result)
    normalized.setdefault("B1", B / math.sqrt(2.0))

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


def build_excavation_animation(
    *,
    H: float,
    B: float,
    q: float,
    Su: float,
    gamma: float,
    animate: bool,
) -> str:
    """
    Build a dimensioned SVG section and conceptual basal-heave animation.

    The motion is illustrative only and does not represent calculated displacement.
    """
    canvas_width = 1120
    canvas_height = 620

    ground_y = 112.0
    excavation_height = 265.0
    base_y = ground_y + excavation_height

    visual_ratio = B / H
    excavation_width = max(190.0, min(520.0, 270.0 * visual_ratio))

    centre_x = 625.0
    left_x = centre_x - excavation_width / 2.0
    right_x = centre_x + excavation_width / 2.0

    b1 = B / math.sqrt(2.0)
    b1_px = excavation_width / math.sqrt(2.0)
    shear_x = max(82.0, left_x - b1_px)
    effective_b1_px = left_x - shear_x

    arc_bottom_y = min(545.0, base_y + 0.72 * effective_b1_px)
    wedge_end_y = min(545.0, base_y + effective_b1_px)

    animation_class = "is-animated" if animate else ""

    strut_ys = [
        ground_y + excavation_height * 0.22,
        ground_y + excavation_height * 0.48,
        ground_y + excavation_height * 0.74,
    ]
    struts = "".join(
        f'''
        <g class="strut">
          <line x1="{left_x + 8:.1f}" y1="{y:.1f}"
                x2="{right_x - 8:.1f}" y2="{y:.1f}" />
          <circle cx="{left_x + 8:.1f}" cy="{y:.1f}" r="4.5" />
          <circle cx="{right_x - 8:.1f}" cy="{y:.1f}" r="4.5" />
        </g>
        '''
        for y in strut_ys
    )

    surcharge_arrows = []
    arrow_positions = [
        shear_x + 22,
        shear_x + 58,
        left_x - 20,
        right_x + 28,
        right_x + 66,
        right_x + 104,
    ]
    for x in arrow_positions:
        surcharge_arrows.append(
            f'''
            <g class="load-arrow">
              <line x1="{x:.1f}" y1="44" x2="{x:.1f}" y2="88"
                    marker-end="url(#loadArrow)" />
              <line x1="{x - 11:.1f}" y1="44" x2="{x + 11:.1f}" y2="44" />
            </g>
            '''
        )
    surcharge_arrows_html = "".join(surcharge_arrows)

    return f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <style>
      html, body {{
        margin: 0;
        padding: 0;
        background: transparent;
        font-family: "Segoe UI", Arial, sans-serif;
        color: #34464f;
      }}

      .diagram-card {{
        background: linear-gradient(180deg, #fffdfa 0%, #faf7f1 100%);
        border: 1px solid #e4ddd2;
        border-radius: 18px;
        padding: 14px 16px 10px;
        box-shadow: 0 10px 28px rgba(73, 86, 89, 0.08);
      }}

      svg {{
        width: 100%;
        height: auto;
        display: block;
      }}

      .soil {{ fill: #e7d4b8; }}
      .excavation {{ fill: #fffdfa; }}

      .wall {{
        stroke: #6c8790;
        stroke-width: 9;
        stroke-linecap: round;
      }}

      .ground-line {{
        stroke: #596d73;
        stroke-width: 3;
      }}

      .strut line {{
        stroke: #8ca9a4;
        stroke-width: 8;
        stroke-linecap: round;
      }}
      .strut circle {{ fill: #6f938d; }}

      .failure-path {{
        fill: none;
        stroke: #c98784;
        stroke-width: 4;
        stroke-dasharray: 12 9;
        stroke-linecap: round;
      }}

      .wedge-line {{
        fill: none;
        stroke: #b99974;
        stroke-width: 2.5;
        stroke-dasharray: 7 6;
      }}

      .retained-block {{
        fill: #dfc4a2;
        fill-opacity: 0.54;
        stroke: #c7a77d;
        stroke-width: 2;
        stroke-dasharray: 7 6;
        transform-box: fill-box;
        transform-origin: center;
      }}

      .heave-zone {{
        fill: #c7ddd3;
        fill-opacity: 0.82;
        stroke: #7aa397;
        stroke-width: 2.5;
        transform-box: fill-box;
        transform-origin: center;
      }}

      .heave-surface {{
        fill: none;
        stroke: #638e83;
        stroke-width: 4;
        stroke-linecap: round;
      }}

      .movement-arrow {{
        stroke: #b86f6c;
        stroke-width: 4;
        stroke-linecap: round;
        fill: none;
      }}

      .load-arrow {{
        stroke: #879aa0;
        stroke-width: 2.5;
        fill: none;
      }}

      .dimension {{
        stroke: #607981;
        stroke-width: 2;
        fill: none;
      }}

      .extension {{
        stroke: #9aabae;
        stroke-width: 1.5;
      }}

      .dimension-text {{
        fill: #40555d;
        font-size: 17px;
        font-weight: 700;
      }}

      .label {{
        fill: #52676e;
        font-size: 16px;
        font-weight: 650;
      }}

      .small-label {{
        fill: #6b7d81;
        font-size: 14px;
      }}

      .value-box {{
        fill: #f3eee6;
        stroke: #ddd2c3;
        stroke-width: 1.5;
      }}

      .value-title {{
        fill: #708185;
        font-size: 13px;
        font-weight: 650;
      }}

      .value-number {{
        fill: #344a52;
        font-size: 18px;
        font-weight: 750;
      }}

      .legend-box {{
        fill: rgba(255, 253, 249, 0.94);
        stroke: #ded6ca;
      }}

      .is-animated .retained-block {{
        animation: soilSink 3.8s ease-in-out infinite;
      }}

      .is-animated .heave-zone,
      .is-animated .heave-surface {{
        animation: baseHeave 3.8s ease-in-out infinite;
      }}

      .is-animated .failure-path {{
        animation: pathFlow 1.5s linear infinite;
      }}

      .is-animated .movement-arrow {{
        animation: arrowPulse 1.9s ease-in-out infinite;
      }}

      .is-animated .load-arrow {{
        animation: loadPulse 2.1s ease-in-out infinite;
      }}

      @keyframes soilSink {{
        0%, 18%, 100% {{ transform: translateY(0); }}
        48%, 72% {{ transform: translateY(15px); }}
      }}

      @keyframes baseHeave {{
        0%, 18%, 100% {{ transform: translateY(0); }}
        48%, 72% {{ transform: translateY(-17px); }}
      }}

      @keyframes pathFlow {{
        from {{ stroke-dashoffset: 0; }}
        to {{ stroke-dashoffset: -42; }}
      }}

      @keyframes arrowPulse {{
        0%, 100% {{ opacity: 0.35; }}
        50% {{ opacity: 1; }}
      }}

      @keyframes loadPulse {{
        0%, 100% {{ opacity: 0.52; }}
        50% {{ opacity: 1; }}
      }}

      @media (prefers-reduced-motion: reduce) {{
        * {{ animation: none !important; }}
      }}
    </style>
    </head>
    <body>
      <div class="diagram-card {animation_class}">
        <svg viewBox="0 0 {canvas_width} {canvas_height}"
             role="img"
             aria-label="Dimensioned excavation section with conceptual basal-heave failure animation">

          <defs>
            <pattern id="soilDots" width="22" height="22" patternUnits="userSpaceOnUse">
              <circle cx="5" cy="6" r="1.7" fill="#c6ab87" opacity="0.45"/>
              <circle cx="16" cy="15" r="1.4" fill="#bfa17d" opacity="0.35"/>
            </pattern>

            <marker id="dimArrow" markerWidth="9" markerHeight="9"
                    refX="4.5" refY="4.5" orient="auto-start-reverse">
              <path d="M0,0 L9,4.5 L0,9 Z" fill="#607981"/>
            </marker>

            <marker id="loadArrow" markerWidth="9" markerHeight="9"
                    refX="4.5" refY="7.5" orient="auto">
              <path d="M0,0 L9,4.5 L0,9 Z" fill="#879aa0"/>
            </marker>

            <marker id="moveArrow" markerWidth="10" markerHeight="10"
                    refX="5" refY="8" orient="auto">
              <path d="M0,0 L10,5 L0,10 Z" fill="#b86f6c"/>
            </marker>
          </defs>

          <rect class="soil" x="30" y="{ground_y:.1f}" width="1060" height="465" rx="5"/>
          <rect x="30" y="{ground_y:.1f}" width="1060" height="465"
                fill="url(#soilDots)" opacity="0.42"/>

          <rect class="excavation"
                x="{left_x:.1f}" y="{ground_y - 3:.1f}"
                width="{excavation_width:.1f}" height="{excavation_height + 4:.1f}"/>

          <path class="retained-block"
                d="M {shear_x:.1f},{ground_y:.1f}
                   L {left_x:.1f},{ground_y:.1f}
                   L {left_x:.1f},{base_y:.1f}
                   L {shear_x:.1f},{wedge_end_y:.1f}
                   Z"/>

          <g>
            <path class="heave-zone"
                  d="M {left_x:.1f},{base_y:.1f}
                     Q {centre_x:.1f},{base_y - 16:.1f} {right_x:.1f},{base_y:.1f}
                     L {right_x:.1f},{base_y + 70:.1f}
                     Q {centre_x:.1f},{base_y + 130:.1f} {left_x:.1f},{base_y + 70:.1f}
                     Z"/>
            <path class="heave-surface"
                  d="M {left_x + 8:.1f},{base_y:.1f}
                     Q {centre_x:.1f},{base_y - 16:.1f} {right_x - 8:.1f},{base_y:.1f}"/>
          </g>

          <path class="failure-path"
                d="M {shear_x:.1f},{ground_y:.1f}
                   L {shear_x:.1f},{wedge_end_y:.1f}
                   Q {shear_x + effective_b1_px * 0.12:.1f},{arc_bottom_y:.1f}
                     {left_x - effective_b1_px * 0.23:.1f},{arc_bottom_y:.1f}
                   Q {left_x + effective_b1_px * 0.10:.1f},{base_y + effective_b1_px * 0.35:.1f}
                     {left_x:.1f},{base_y:.1f}"/>

          <line class="wedge-line"
                x1="{left_x:.1f}" y1="{base_y:.1f}"
                x2="{shear_x:.1f}" y2="{wedge_end_y:.1f}"/>

          <line class="wall" x1="{left_x:.1f}" y1="{ground_y:.1f}"
                x2="{left_x:.1f}" y2="{base_y + 48:.1f}"/>
          <line class="wall" x1="{right_x:.1f}" y1="{ground_y:.1f}"
                x2="{right_x:.1f}" y2="{base_y + 48:.1f}"/>
          {struts}

          <line class="ground-line" x1="30" y1="{ground_y:.1f}"
                x2="{left_x:.1f}" y2="{ground_y:.1f}"/>
          <line class="ground-line" x1="{right_x:.1f}" y1="{ground_y:.1f}"
                x2="1090" y2="{ground_y:.1f}"/>

          {surcharge_arrows_html}
          <text class="label" x="{right_x + 123:.1f}" y="68">q = {q:.1f} kPa</text>

          <line class="movement-arrow"
                x1="{(shear_x + left_x) / 2:.1f}" y1="{ground_y + 85:.1f}"
                x2="{(shear_x + left_x) / 2:.1f}" y2="{ground_y + 145:.1f}"
                marker-end="url(#moveArrow)"/>
          <text class="small-label"
                x="{(shear_x + left_x) / 2 - 42:.1f}" y="{ground_y + 170:.1f}">
            ground settlement
          </text>

          <line class="movement-arrow"
                x1="{centre_x:.1f}" y1="{base_y + 72:.1f}"
                x2="{centre_x:.1f}" y2="{base_y + 8:.1f}"
                marker-end="url(#moveArrow)"/>
          <text class="small-label"
                x="{centre_x + 16:.1f}" y="{base_y + 48:.1f}">
            basal heave
          </text>

          <line class="extension" x1="{left_x:.1f}" y1="{ground_y - 6:.1f}"
                x2="{left_x:.1f}" y2="28"/>
          <line class="extension" x1="{right_x:.1f}" y1="{ground_y - 6:.1f}"
                x2="{right_x:.1f}" y2="28"/>
          <line class="dimension" x1="{left_x:.1f}" y1="28"
                x2="{right_x:.1f}" y2="28"
                marker-start="url(#dimArrow)" marker-end="url(#dimArrow)"/>
          <text class="dimension-text" text-anchor="middle"
                x="{centre_x:.1f}" y="21">B = {B:.2f} m</text>

          <line class="extension" x1="{shear_x:.1f}" y1="{ground_y - 5:.1f}"
                x2="{shear_x:.1f}" y2="83"/>
          <line class="extension" x1="{left_x:.1f}" y1="{ground_y - 5:.1f}"
                x2="{left_x:.1f}" y2="83"/>
          <line class="dimension" x1="{shear_x:.1f}" y1="83"
                x2="{left_x:.1f}" y2="83"
                marker-start="url(#dimArrow)" marker-end="url(#dimArrow)"/>
          <text class="dimension-text" text-anchor="middle"
                x="{(shear_x + left_x) / 2:.1f}" y="76">
            B₁ = B/√2 = {b1:.2f} m
          </text>

          <line class="extension" x1="{right_x + 7:.1f}" y1="{ground_y:.1f}"
                x2="{right_x + 82:.1f}" y2="{ground_y:.1f}"/>
          <line class="extension" x1="{right_x + 7:.1f}" y1="{base_y:.1f}"
                x2="{right_x + 82:.1f}" y2="{base_y:.1f}"/>
          <line class="dimension" x1="{right_x + 67:.1f}" y1="{ground_y:.1f}"
                x2="{right_x + 67:.1f}" y2="{base_y:.1f}"
                marker-start="url(#dimArrow)" marker-end="url(#dimArrow)"/>
          <text class="dimension-text"
                transform="translate({right_x + 92:.1f},{(ground_y + base_y) / 2:.1f}) rotate(90)"
                text-anchor="middle">H = {H:.2f} m</text>

          <path class="dimension"
                d="M {left_x - 34:.1f},{base_y:.1f}
                   A 34,34 0 0 1 {left_x - 10:.1f},{base_y + 24:.1f}"/>
          <text class="small-label"
                x="{left_x - 57:.1f}" y="{base_y + 31:.1f}">45°</text>

          <rect class="value-box" x="818" y="458" width="252" height="102" rx="13"/>
          <text class="value-title" x="838" y="484">SOIL PARAMETERS</text>
          <text class="value-number" x="838" y="515">Sᵤ = {Su:.1f} kPa</text>
          <text class="value-number" x="838" y="544">γ = {gamma:.1f} kN/m³</text>

          <rect class="legend-box" x="48" y="474" width="305" height="86" rx="12"/>
          <line class="failure-path" x1="66" y1="500" x2="112" y2="500"/>
          <text class="small-label" x="126" y="505">conceptual failure surface</text>
          <rect class="heave-zone" x="66" y="520" width="46" height="20" rx="5"/>
          <text class="small-label" x="126" y="536">soil moving toward excavation</text>

          <text class="label" text-anchor="middle"
                x="{centre_x:.1f}" y="{ground_y + 25:.1f}">BRACED EXCAVATION</text>
        </svg>
      </div>
    </body>
    </html>
    '''


st.set_page_config(
    page_title="Basal Heave Stability Calculator",
    page_icon="🏗️",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #34464f;
        --muted: #708185;
        --line: #e4ddd2;
        --cream: #fffdfa;
        --sand: #f3eee6;
        --sage: #8ca9a4;
    }

    .stApp {
        background:
          radial-gradient(circle at 8% 0%, rgba(231, 212, 184, 0.23), transparent 27%),
          radial-gradient(circle at 98% 8%, rgba(199, 221, 211, 0.25), transparent 26%),
          #f8f6f1;
        color: var(--ink);
    }

    .block-container {
        max-width: 1480px;
        padding-top: 1.35rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: #34464f !important;
        letter-spacing: -0.015em;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(180deg, #fffdfa 0%, #f7f2ea 100%);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 15px 17px;
        box-shadow: 0 8px 23px rgba(73, 86, 89, 0.06);
    }

    [data-testid="stMetricLabel"] { color: #708185; }
    [data-testid="stMetricValue"] { color: #354c54; }

    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        background-color: #fffdfa;
        border-color: #ddd4c8;
        border-radius: 11px;
    }

    .stDownloadButton > button {
        background: #789b95;
        color: white;
        border: 0;
        border-radius: 11px;
        padding: 0.58rem 1rem;
        font-weight: 700;
    }

    .stDownloadButton > button:hover {
        background: #63877f;
        color: white;
        border: 0;
    }

    div[data-testid="stAlert"] { border-radius: 13px; }

    .scope-box {
        background: #f3eee6;
        border-left: 5px solid #8ca9a4;
        border-radius: 11px;
        padding: 0.85rem 1rem;
        margin: 0.5rem 0 1rem 0;
        color: #40545c;
    }

    .method-note {
        background: #fffdfa;
        border: 1px solid #e4ddd2;
        border-radius: 13px;
        padding: 0.85rem 1rem;
        color: #586c72;
        margin-top: 0.75rem;
    }

    hr { border-color: #e2dbd0 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

engine = get_engine()

st.title("Basal Heave Stability Calculator")
st.caption(
    "Terzaghi and Bjerrum & Eide calculated directly from equations • "
    "Only Optum results are interpolated from the numerical database"
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
        "The database is used only for interpolation of the Optum lower and upper "
        "bounds. Terzaghi and Bjerrum & Eide are recalculated analytically."
    )
    st.write(
        "The absent Optum corner is q = 20 kPa with Su = 5 kPa. "
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

settings_cols = st.columns([1.55, 1.0, 1.2, 1.6])
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
with settings_cols[2]:
    animate_mechanism = st.toggle(
        "Animate mechanism",
        value=True,
        help="Conceptual animation only; it is not a displacement prediction.",
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
main_results[0].metric(
    "Terzaghi F.S. — equation",
    format_fs(result["terzaghi"]),
)
main_results[1].metric(
    "Bjerrum & Eide F.S. — equation",
    f"{result['bjerrum']:.3f}",
)
main_results[2].metric(
    "Recommended Optum F.S. — interpolation",
    f"{result['recommended']:.3f}",
)

if result["terzaghi"] is None:
    st.warning(
        "The Terzaghi denominator is zero or negative for this input: "
        f"γH + q − SuH/B₁ = {result['terzaghi_denominator']:.3f} kPa. "
        "A positive Terzaghi factor of safety is therefore not reported."
    )

optum_results = st.columns(4)
optum_results[0].metric(
    "Optum lower bound",
    f"{result['numerical_lower']:.3f}",
)
optum_results[1].metric(
    "Optum average",
    f"{result['numerical_average']:.3f}",
)
optum_results[2].metric(
    "Optum upper bound",
    f"{result['numerical_upper']:.3f}",
)
optum_results[3].metric(
    "Upper–lower gap",
    f"{result['bound_gap_percent']:.1f}%",
)

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
st.dataframe(
    calculation_details,
    use_container_width=True,
    hide_index=True,
)

st.subheader("Excavation section and conceptual basal-heave mechanism")
components.html(
    build_excavation_animation(
        H=H,
        B=B,
        q=q,
        Su=Su,
        gamma=gamma,
        animate=animate_mechanism,
    ),
    height=675,
    scrolling=False,
)

st.markdown(
    """
    <div class="method-note">
    <b>Animation note:</b> the movement is a conceptual illustration of the
    basal-heave mechanism. It shows settlement of the retained ground and
    upward movement of the excavation base; it is not a calculated displacement,
    deformation contour, or time-history prediction.
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader("Nearest original Optum models")
nearest = pd.DataFrame(
    engine.nearest_cases(
        Su=Su,
        gamma=gamma,
        q=q,
        H=H,
        B=B,
        count=8,
    )
)
st.dataframe(
    nearest,
    use_container_width=True,
    hide_index=True,
)

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
- Terzaghi and Bjerrum & Eide are calculated directly from equations.
- Optum extrapolation outside the database domain is blocked.
"""
)
