from __future__ import annotations

from pathlib import Path
import csv
import html
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


def basis_label(value: str) -> str:
    return {
        "lower": "Lower bound — conservative",
        "average": "Average of lower and upper bounds",
        "upper": "Upper bound",
    }[value]


def format_fs(value: float | None) -> str:
    return "Not valid" if value is None else f"{value:.3f}"


def resolved_mechanism(selection: str, h_over_b: float) -> str:
    if selection == "Auto from H/B":
        return "Terzaghi" if h_over_b <= 1.0 else "Bjerrum & Eide"
    return selection


def mechanism_subtitle(mechanism: str, h_over_b: float) -> str:
    if mechanism == "Terzaghi":
        return (
            f"Wide-excavation visualization • H/B = {h_over_b:.2f} • "
            "failure zone extends toward the retained ground"
        )
    return (
        f"Deep/narrow-excavation visualization • H/B = {h_over_b:.2f} • "
        "localized basal failure below the excavation"
    )


def excavation_animation_html(
    *,
    H: float,
    B: float,
    q: float,
    Su: float,
    gamma: float,
    mechanism: str,
    animate: bool,
) -> str:
    """Return a responsive SVG illustration of the excavation and basal failure."""

    view_w = 1100.0
    view_h = 650.0
    ground_y = 145.0

    # Fit the entered dimensions into the drawing while retaining the correct H/B ratio.
    max_width = 430.0
    max_depth = 320.0
    scale = min(max_width / B, max_depth / H)
    excavation_width = B * scale
    excavation_depth = H * scale

    centre_x = 560.0
    left_wall = centre_x - excavation_width / 2.0
    right_wall = centre_x + excavation_width / 2.0
    base_y = ground_y + excavation_depth

    wall_thickness = 10.0
    embedment = min(68.0, max(35.0, 0.18 * excavation_depth))
    wall_toe_y = base_y + embedment

    # Bracing positions are schematic, because support spacing is not an app input.
    strut_levels = [
        ground_y + excavation_depth * 0.22,
        ground_y + excavation_depth * 0.50,
        ground_y + excavation_depth * 0.78,
    ]

    # Animation classes are turned off when the user pauses the mechanism.
    animation_state = "running" if animate else "paused"
    dash_duration = "2.8s" if mechanism == "Terzaghi" else "2.2s"

    if mechanism == "Terzaghi":
        # Large mechanism reaching the retained ground on both sides.
        left_start_x = max(70.0, left_wall - max(145.0, excavation_depth * 0.60))
        right_start_x = min(1030.0, right_wall + max(145.0, excavation_depth * 0.60))
        bulb_y = min(590.0, wall_toe_y + max(75.0, excavation_width * 0.20))

        left_path = (
            f"M {left_start_x:.1f} {ground_y:.1f} "
            f"L {left_start_x:.1f} {base_y + 25:.1f} "
            f"Q {left_wall - 55:.1f} {bulb_y:.1f}, {centre_x:.1f} {bulb_y:.1f}"
        )
        right_path = (
            f"M {right_start_x:.1f} {ground_y:.1f} "
            f"L {right_start_x:.1f} {base_y + 25:.1f} "
            f"Q {right_wall + 55:.1f} {bulb_y:.1f}, {centre_x:.1f} {bulb_y:.1f}"
        )
        mechanism_note = "Terzaghi-type global mechanism"
        side_arrow_dx = 18
        side_arrow_dy = 18
    else:
        # Localized bearing-capacity bulbs beneath a deep/narrow excavation.
        bulb_y = min(590.0, wall_toe_y + max(70.0, excavation_width * 0.24))
        left_path = (
            f"M {left_wall:.1f} {base_y:.1f} "
            f"C {left_wall - 65:.1f} {base_y + 35:.1f}, "
            f"{left_wall - 62:.1f} {bulb_y:.1f}, {centre_x:.1f} {bulb_y:.1f}"
        )
        right_path = (
            f"M {right_wall:.1f} {base_y:.1f} "
            f"C {right_wall + 65:.1f} {base_y + 35:.1f}, "
            f"{right_wall + 62:.1f} {bulb_y:.1f}, {centre_x:.1f} {bulb_y:.1f}"
        )
        mechanism_note = "Bjerrum & Eide-type local mechanism"
        side_arrow_dx = 10
        side_arrow_dy = 10

    # A soft, professional pastel palette.
    palette = {
        "background": "#FFFDFC",
        "card": "#FFF8F3",
        "soil": "#EED8C8",
        "soil_dark": "#DEC0AB",
        "excavation": "#FFFDFB",
        "wall": "#59636B",
        "strut": "#8EA2AE",
        "text": "#604E4A",
        "muted": "#8D7770",
        "dimension": "#88706A",
        "failure": "#D97968",
        "failure_fill": "#F4B7AA",
        "heave": "#D9917F",
        "arrow": "#A86558",
        "grid": "#EADFD8",
        "accent": "#B68779",
    }

    def arrows(side: str) -> str:
        if side == "left":
            x = left_wall - 105
            transform = f"translate({side_arrow_dx}px,{side_arrow_dy}px)"
            line_x2 = x + 34
        else:
            x = right_wall + 105
            transform = f"translate(-{side_arrow_dx}px,{side_arrow_dy}px)"
            line_x2 = x - 34

        y_positions = [ground_y + 55, ground_y + 105, ground_y + 155]
        output: list[str] = []
        for y in y_positions:
            if y >= base_y - 10:
                continue
            if side == "left":
                marker = "url(#arrow-right)"
            else:
                marker = "url(#arrow-left)"
            output.append(
                f'<line class="soil-motion" x1="{x:.1f}" y1="{y:.1f}" '
                f'x2="{line_x2:.1f}" y2="{y + 16:.1f}" '
                f'stroke="{palette["arrow"]}" stroke-width="4" marker-end="{marker}" '
                f'style="--motion-transform:{transform};" />'
            )
        return "".join(output)

    struts_svg = "".join(
        f"""
        <line x1="{left_wall + wall_thickness:.1f}" y1="{level:.1f}"
              x2="{right_wall - wall_thickness:.1f}" y2="{level:.1f}"
              stroke="{palette['strut']}" stroke-width="8" stroke-linecap="round"/>
        <circle cx="{left_wall + wall_thickness:.1f}" cy="{level:.1f}" r="6" fill="{palette['wall']}"/>
        <circle cx="{right_wall - wall_thickness:.1f}" cy="{level:.1f}" r="6" fill="{palette['wall']}"/>
        """
        for level in strut_levels
    )

    surcharge_arrows: list[str] = []
    for x in [left_wall - 150, left_wall - 95, left_wall - 40, right_wall + 40, right_wall + 95, right_wall + 150]:
        surcharge_arrows.append(
            f"""
            <line x1="{x:.1f}" y1="70" x2="{x:.1f}" y2="118"
                  stroke="{palette['accent']}" stroke-width="3" marker-end="url(#arrow-down)"/>
            """
        )

    title = html.escape(mechanism_note)
    animation_label = "Animated" if animate else "Paused"

    return f"""
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <style>
        html, body {{ margin: 0; padding: 0; background: transparent; }}
        * {{ box-sizing: border-box; }}
        .visual-card {{
          font-family: Inter, "Segoe UI", Arial, sans-serif;
          background: linear-gradient(145deg, {palette['background']} 0%, {palette['card']} 100%);
          border: 1px solid {palette['grid']};
          border-radius: 22px;
          padding: 18px 20px 14px;
          box-shadow: 0 12px 34px rgba(96, 78, 74, 0.08);
        }}
        .visual-head {{
          display: flex; justify-content: space-between; align-items: flex-start;
          gap: 14px; margin: 0 4px 7px;
        }}
        .visual-title {{ color: {palette['text']}; font-size: 22px; font-weight: 760; }}
        .visual-subtitle {{ color: {palette['muted']}; font-size: 13px; margin-top: 4px; }}
        .status-pill {{
          color: {palette['text']}; background: #F5E5DC; border: 1px solid #E9CFC2;
          border-radius: 999px; padding: 6px 10px; font-size: 12px; font-weight: 700;
          white-space: nowrap;
        }}
        svg {{ width: 100%; height: auto; display: block; }}
        .failure-line {{
          stroke-dasharray: 14 10;
          animation: dash {dash_duration} linear infinite;
          animation-play-state: {animation_state};
        }}
        .heave-block {{
          transform-origin: {centre_x:.1f}px {base_y:.1f}px;
          animation: heave 1.7s ease-in-out infinite;
          animation-play-state: {animation_state};
        }}
        .soil-motion {{
          animation: soilMove 1.7s ease-in-out infinite;
          animation-play-state: {animation_state};
        }}
        .pulse-zone {{
          animation: pulse 1.7s ease-in-out infinite;
          animation-play-state: {animation_state};
        }}
        @keyframes dash {{ to {{ stroke-dashoffset: -96; }} }}
        @keyframes heave {{
          0%, 100% {{ transform: translateY(0); }}
          50% {{ transform: translateY(-12px); }}
        }}
        @keyframes soilMove {{
          0%, 100% {{ transform: translate(0,0); opacity: .72; }}
          50% {{ transform: var(--motion-transform); opacity: 1; }}
        }}
        @keyframes pulse {{
          0%, 100% {{ opacity: .16; }}
          50% {{ opacity: .36; }}
        }}
        .footer-note {{
          color: {palette['muted']}; font-size: 12px; line-height: 1.45;
          margin: 4px 8px 0;
        }}
      </style>
    </head>
    <body>
      <div class="visual-card">
        <div class="visual-head">
          <div>
            <div class="visual-title">Excavation section & basal-heave mechanism</div>
            <div class="visual-subtitle">{title} • Dimensions update automatically from the given inputs</div>
          </div>
          <div class="status-pill">{animation_label}</div>
        </div>

        <svg viewBox="0 0 {view_w:.0f} {view_h:.0f}" role="img" aria-label="Basal heave failure mechanism">
          <defs>
            <marker id="arrow-down" markerWidth="10" markerHeight="10" refX="5" refY="8" orient="auto">
              <path d="M1,1 L5,8 L9,1" fill="none" stroke="{palette['accent']}" stroke-width="2"/>
            </marker>
            <marker id="arrow-up" markerWidth="10" markerHeight="10" refX="5" refY="2" orient="auto">
              <path d="M1,9 L5,2 L9,9" fill="none" stroke="{palette['arrow']}" stroke-width="2"/>
            </marker>
            <marker id="arrow-right" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
              <path d="M1,1 L8,5 L1,9" fill="none" stroke="{palette['arrow']}" stroke-width="2"/>
            </marker>
            <marker id="arrow-left" markerWidth="10" markerHeight="10" refX="2" refY="5" orient="auto">
              <path d="M9,1 L2,5 L9,9" fill="none" stroke="{palette['arrow']}" stroke-width="2"/>
            </marker>
            <marker id="dim-start" markerWidth="8" markerHeight="8" refX="1" refY="4" orient="auto">
              <path d="M7,1 L1,4 L7,7" fill="none" stroke="{palette['dimension']}" stroke-width="1.6"/>
            </marker>
            <marker id="dim-end" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
              <path d="M1,1 L7,4 L1,7" fill="none" stroke="{palette['dimension']}" stroke-width="1.6"/>
            </marker>
            <pattern id="soil-pattern" width="22" height="22" patternUnits="userSpaceOnUse">
              <rect width="22" height="22" fill="{palette['soil']}"/>
              <circle cx="5" cy="7" r="1.6" fill="{palette['soil_dark']}" opacity=".65"/>
              <circle cx="16" cy="15" r="1.3" fill="{palette['soil_dark']}" opacity=".52"/>
            </pattern>
          </defs>

          <!-- Soil and ground -->
          <rect x="25" y="{ground_y:.1f}" width="{left_wall - 25:.1f}" height="{view_h - ground_y - 25:.1f}"
                rx="8" fill="url(#soil-pattern)"/>
          <rect x="{right_wall:.1f}" y="{ground_y:.1f}" width="{view_w - right_wall - 25:.1f}"
                height="{view_h - ground_y - 25:.1f}" rx="8" fill="url(#soil-pattern)"/>
          <rect x="{left_wall:.1f}" y="{base_y:.1f}" width="{excavation_width:.1f}"
                height="{view_h - base_y - 25:.1f}" fill="url(#soil-pattern)"/>
          <line x1="25" y1="{ground_y:.1f}" x2="{view_w - 25:.1f}" y2="{ground_y:.1f}"
                stroke="{palette['dimension']}" stroke-width="3"/>

          <!-- Surcharge -->
          {''.join(surcharge_arrows)}
          <text x="{left_wall - 105:.1f}" y="55" fill="{palette['text']}" font-size="20" font-weight="700">q = {q:.1f} kPa</text>
          <text x="{right_wall + 30:.1f}" y="55" fill="{palette['muted']}" font-size="15">uniform surcharge</text>

          <!-- Excavation void -->
          <rect x="{left_wall:.1f}" y="{ground_y:.1f}" width="{excavation_width:.1f}" height="{excavation_depth:.1f}"
                fill="{palette['excavation']}"/>
          <line x1="{left_wall:.1f}" y1="{base_y:.1f}" x2="{right_wall:.1f}" y2="{base_y:.1f}"
                stroke="{palette['soil_dark']}" stroke-width="3"/>

          <!-- Retaining walls and bracing -->
          <rect x="{left_wall - wall_thickness / 2:.1f}" y="{ground_y - 4:.1f}" width="{wall_thickness:.1f}"
                height="{excavation_depth + embedment + 4:.1f}" rx="3" fill="{palette['wall']}"/>
          <rect x="{right_wall - wall_thickness / 2:.1f}" y="{ground_y - 4:.1f}" width="{wall_thickness:.1f}"
                height="{excavation_depth + embedment + 4:.1f}" rx="3" fill="{palette['wall']}"/>
          {struts_svg}

          <!-- Failure zone and animated surface -->
          <path class="pulse-zone" d="{left_path} L {centre_x:.1f} {base_y:.1f} Z"
                fill="{palette['failure_fill']}" stroke="none"/>
          <path class="pulse-zone" d="{right_path} L {centre_x:.1f} {base_y:.1f} Z"
                fill="{palette['failure_fill']}" stroke="none"/>
          <path class="failure-line" d="{left_path}" fill="none" stroke="{palette['failure']}" stroke-width="5" stroke-linecap="round"/>
          <path class="failure-line" d="{right_path}" fill="none" stroke="{palette['failure']}" stroke-width="5" stroke-linecap="round"/>

          <!-- Heaving base block -->
          <g class="heave-block">
            <path d="M {left_wall + excavation_width * 0.22:.1f} {base_y + 7:.1f}
                     Q {centre_x:.1f} {base_y - 7:.1f}, {right_wall - excavation_width * 0.22:.1f} {base_y + 7:.1f}
                     L {right_wall - excavation_width * 0.27:.1f} {base_y + 32:.1f}
                     Q {centre_x:.1f} {base_y + 20:.1f}, {left_wall + excavation_width * 0.27:.1f} {base_y + 32:.1f} Z"
                  fill="{palette['heave']}" opacity=".72"/>
            <line x1="{centre_x:.1f}" y1="{base_y + 20:.1f}" x2="{centre_x:.1f}" y2="{base_y - 58:.1f}"
                  stroke="{palette['arrow']}" stroke-width="5" marker-end="url(#arrow-up)"/>
          </g>

          <!-- Soil movement arrows -->
          {arrows('left')}
          {arrows('right')}

          <!-- B dimension -->
          <line x1="{left_wall:.1f}" y1="105" x2="{right_wall:.1f}" y2="105"
                stroke="{palette['dimension']}" stroke-width="2"
                marker-start="url(#dim-start)" marker-end="url(#dim-end)"/>
          <line x1="{left_wall:.1f}" y1="96" x2="{left_wall:.1f}" y2="{ground_y - 4:.1f}"
                stroke="{palette['dimension']}" stroke-width="1.5"/>
          <line x1="{right_wall:.1f}" y1="96" x2="{right_wall:.1f}" y2="{ground_y - 4:.1f}"
                stroke="{palette['dimension']}" stroke-width="1.5"/>
          <rect x="{centre_x - 57:.1f}" y="80" width="114" height="33" rx="16" fill="{palette['card']}" stroke="{palette['grid']}"/>
          <text x="{centre_x:.1f}" y="102" text-anchor="middle" fill="{palette['text']}" font-size="18" font-weight="750">B = {B:.2f} m</text>

          <!-- H dimension -->
          <line x1="{left_wall - 72:.1f}" y1="{ground_y:.1f}" x2="{left_wall - 72:.1f}" y2="{base_y:.1f}"
                stroke="{palette['dimension']}" stroke-width="2"
                marker-start="url(#dim-start)" marker-end="url(#dim-end)"/>
          <line x1="{left_wall - 80:.1f}" y1="{ground_y:.1f}" x2="{left_wall - 8:.1f}" y2="{ground_y:.1f}"
                stroke="{palette['dimension']}" stroke-width="1.5"/>
          <line x1="{left_wall - 80:.1f}" y1="{base_y:.1f}" x2="{left_wall - 8:.1f}" y2="{base_y:.1f}"
                stroke="{palette['dimension']}" stroke-width="1.5"/>
          <g transform="translate({left_wall - 102:.1f},{(ground_y + base_y) / 2:.1f}) rotate(-90)">
            <rect x="-57" y="-17" width="114" height="34" rx="17" fill="{palette['card']}" stroke="{palette['grid']}"/>
            <text x="0" y="6" text-anchor="middle" fill="{palette['text']}" font-size="18" font-weight="750">H = {H:.2f} m</text>
          </g>

          <!-- Parameter badges -->
          <g transform="translate(35,585)">
            <rect width="1030" height="44" rx="18" fill="#FFF9F5" stroke="{palette['grid']}"/>
            <text x="24" y="28" fill="{palette['text']}" font-size="16" font-weight="700">Su = {Su:.1f} kPa</text>
            <text x="205" y="28" fill="{palette['text']}" font-size="16" font-weight="700">γ = {gamma:.1f} kN/m³</text>
            <text x="430" y="28" fill="{palette['text']}" font-size="16" font-weight="700">H/B = {H / B:.3f}</text>
            <text x="620" y="28" fill="{palette['text']}" font-size="16" font-weight="700">B/H = {B / H:.3f}</text>
            <text x="810" y="28" fill="{palette['failure']}" font-size="16" font-weight="750">{html.escape(mechanism_note)}</text>
          </g>
        </svg>
        <div class="footer-note">
          Conceptual mechanism visualization for communication and interpretation. The dimensions are drawn to the entered H/B ratio;
          the animation is not a displacement prediction from Optum.
        </div>
      </div>
    </body>
    </html>
    """


st.set_page_config(
    page_title="Basal Heave Stability Calculator",
    page_icon="🏗️",
    layout="wide",
)

# Soft pastel theme for the complete application.
st.markdown(
    """
    <style>
      :root {
        --page: #FCF9F7;
        --card: #FFFDFC;
        --card-soft: #FFF7F2;
        --line: #EADDD5;
        --text: #5D4B47;
        --muted: #8A746E;
        --accent: #B98778;
        --accent-dark: #96685C;
        --success: #5F806F;
        --danger: #B7655B;
      }

      .stApp {
        background:
          radial-gradient(circle at 10% 5%, rgba(240, 219, 207, .28), transparent 28%),
          linear-gradient(180deg, #FFFDFC 0%, var(--page) 100%);
        color: var(--text);
      }
      .block-container {
        max-width: 1500px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
      }
      h1, h2, h3, h4, p, label { color: var(--text); }
      h1 { letter-spacing: -.02em; }
      [data-testid="stCaptionContainer"] { color: var(--muted); }

      [data-testid="stMetric"] {
        background: linear-gradient(145deg, #FFFDFC 0%, #FFF7F2 100%);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 16px 18px;
        box-shadow: 0 8px 24px rgba(93, 75, 71, .06);
      }
      [data-testid="stMetricLabel"] { color: var(--muted); }
      [data-testid="stMetricValue"] { color: var(--text); }

      [data-testid="stExpander"] {
        background: rgba(255, 253, 252, .78);
        border: 1px solid var(--line);
        border-radius: 16px;
        overflow: hidden;
      }
      [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 16px;
        overflow: hidden;
      }
      div[data-baseweb="input"] > div,
      div[data-baseweb="select"] > div {
        background: #FFFDFC;
        border-color: #DDC9BF;
        border-radius: 12px;
      }
      div[data-baseweb="input"] > div:focus-within,
      div[data-baseweb="select"] > div:focus-within {
        border-color: var(--accent);
        box-shadow: 0 0 0 1px var(--accent);
      }
      .stButton > button, .stDownloadButton > button {
        border: 1px solid #D4B8AB;
        border-radius: 12px;
        background: linear-gradient(145deg, #C99A8A, #B98778);
        color: white;
        font-weight: 700;
        box-shadow: 0 6px 16px rgba(150, 104, 92, .16);
      }
      .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: var(--accent-dark);
        background: var(--accent-dark);
        color: white;
      }
      [data-testid="stAlert"] { border-radius: 14px; }
      hr { border-color: var(--line); }

      .scope-box {
        background: linear-gradient(145deg, #FFF9F5, #FFFDFC);
        border: 1px solid var(--line);
        border-left: 5px solid var(--accent);
        border-radius: 14px;
        padding: .9rem 1.1rem;
        margin: .5rem 0 1rem;
        color: var(--text);
      }
      .equation-card {
        background: linear-gradient(145deg, #FFFDFC, #FFF8F4);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 1rem 1.2rem;
        min-height: 255px;
        box-shadow: 0 7px 20px rgba(93, 75, 71, .05);
      }
      .section-note {
        background: #FFF8F4;
        border: 1px solid var(--line);
        border-radius: 14px;
        color: var(--muted);
        padding: 12px 15px;
        line-height: 1.55;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

engine = get_engine()

st.title("Basal Heave Stability Calculator")
st.caption(
    "Terzaghi and Bjerrum & Eide are calculated directly from equations • "
    "Only the Optum lower and upper bounds are interpolated from the numerical database"
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
        "The database is used only for the Optum lower and upper bounds. "
        "The classical methods are recalculated analytically for each input."
    )
    st.write(
        "The only absent Optum corner is q = 20 kPa with Su = 5 kPa. "
        "For 10 < q ≤ 20 kPa, the data-supported boundary is Su ≥ q/2."
    )

st.subheader("Given input parameters")

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
        help=f"The Optum database requires Su ≥ {minimum_su:g} kPa when q = {q:g} kPa.",
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

settings_cols = st.columns([1.40, 1.0, 1.45, 1.0])
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
    mechanism_selection = st.selectbox(
        "Failure-mechanism visualization",
        options=["Auto from H/B", "Terzaghi", "Bjerrum & Eide"],
        index=0,
        help=(
            "Auto displays a Terzaghi-type mechanism for H/B ≤ 1 and a localized "
            "Bjerrum & Eide-type mechanism for H/B > 1."
        ),
    )
with settings_cols[3]:
    animate_mechanism = st.toggle("Animate mechanism", value=True)

try:
    result = engine.calculate(
        Su=Su,
        gamma=gamma,
        q=q,
        H=H,
        B=B,
        recommendation_basis=recommendation_basis,
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
        f"The recommended Optum F.S. is {difference:.3f} above the target of {target_fs:.2f}."
    )
else:
    st.error(
        f"The recommended Optum F.S. is {abs(difference):.3f} below the target of {target_fs:.2f}."
    )

st.subheader("Equations used")

equation_cols = st.columns(2)
with equation_cols[0]:
    with st.container(border=True):
        st.markdown("**Terzaghi (1943)**")
        st.latex(r"B_1=\frac{B}{\sqrt{2}}")
        st.latex(r"F.S._T=\frac{5.7S_u}{\gamma H+q-\dfrac{S_uH}{B_1}}")
        st.caption(
            "Homogeneous clay: Cᵤb = Cᵤh = Su. No nearby hard stratum: B₁ = B/√2."
        )

with equation_cols[1]:
    with st.container(border=True):
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

# The previous family-of-curves plot is intentionally replaced by a dynamic section.
st.subheader("Excavation section and animated failure mechanism")

shown_mechanism = resolved_mechanism(mechanism_selection, result["H_over_B"])
st.caption(mechanism_subtitle(shown_mechanism, result["H_over_B"]))
components.html(
    excavation_animation_html(
        H=H,
        B=B,
        q=q,
        Su=Su,
        gamma=gamma,
        mechanism=shown_mechanism,
        animate=animate_mechanism,
    ),
    height=735,
    scrolling=False,
)
st.markdown(
    """
    <div class="section-note">
      <b>Interpretation:</b> the dashed curves illustrate the assumed basal-failure mechanism,
      the central arrow represents upward heave inside the excavation, and the side arrows
      represent soil movement toward the released excavation. This is a conceptual visualization;
      the Optum factor of safety continues to come from database interpolation.
    </div>
    """,
    unsafe_allow_html=True,
)

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
    ("Visualized mechanism", shown_mechanism),
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
- Surcharge q is included in the driving pressure in both classical equations.
- R_int = 1.0, matching the Optum numerical database.
- Only Optum lower and upper bounds are interpolated from the database.
- Terzaghi and Bjerrum & Eide are calculated directly from equations for every input.
- Optum extrapolation outside the database domain is blocked.
- The animated section is conceptual and is not an Optum displacement output.
    """
)
