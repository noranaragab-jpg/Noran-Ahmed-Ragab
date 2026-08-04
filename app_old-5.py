from __future__ import annotations

from pathlib import Path
import csv
import io
import json
import math

import pandas as pd
import plotly.graph_objects as go
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


def nc_from_ratio(h_over_b: float) -> float:
    """Same piecewise Nc rule used by engine.py."""
    if h_over_b > 3.0:
        return 7.2
    return 5.14 + 1.05 * h_over_b - 0.18 * h_over_b**2


def nc_chart(h_over_b: float, nc: float) -> go.Figure:
    """Create the external Nc chart while keeping the equations behind the UI."""
    x_poly = [i / 100 for i in range(0, 301)]
    y_poly = [nc_from_ratio(x) for x in x_poly]
    x_plateau = [3.0001 + (6.0 - 3.0001) * i / 180 for i in range(181)]
    y_plateau = [7.2] * len(x_plateau)

    shown_x = min(max(h_over_b, 0.0), 6.0)
    shown_nc = nc_from_ratio(h_over_b)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_poly,
            y=y_poly,
            mode="lines",
            name="Nc used for H/B ≤ 3",
            line=dict(color="#7B9FA0", width=4),
            hovertemplate="H/B = %{x:.2f}<br>Nc = %{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_plateau,
            y=y_plateau,
            mode="lines",
            name="Nc = 7.2 for H/B > 3",
            line=dict(color="#C98D82", width=4),
            hovertemplate="H/B = %{x:.2f}<br>Nc = 7.200<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[shown_x],
            y=[shown_nc],
            mode="markers+text",
            name="Selected geometry",
            marker=dict(size=14, color="#8F5F58", line=dict(width=3, color="#FFFDFC")),
            text=[f"  Nc = {nc:.3f}"],
            textposition="middle right",
            textfont=dict(color="#5C4B47", size=13),
            hovertemplate=(
                f"Selected H/B = {h_over_b:.3f}<br>"
                f"Calculated Nc = {nc:.3f}<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=3.0, line_dash="dot", line_color="#BBA9A2", line_width=2)
    fig.add_annotation(
        x=3.0,
        y=5.02,
        text="Piecewise transition",
        showarrow=False,
        textangle=-90,
        font=dict(color="#88736C", size=11),
        xanchor="right",
    )
    fig.add_annotation(
        x=5.85,
        y=7.28,
        text="Plane-strain / infinitely long excavation",
        showarrow=False,
        font=dict(color="#6F625E", size=12),
        xanchor="right",
    )
    fig.update_layout(
        title=dict(
            text="Bjerrum & Eide bearing-capacity factor",
            font=dict(color="#5C4B47", size=19),
            x=0.02,
        ),
        xaxis=dict(
            title="H/B",
            range=[0, 6],
            dtick=1,
            gridcolor="#E9E0DB",
            zeroline=False,
            showline=True,
            linecolor="#CFC0B9",
        ),
        yaxis=dict(
            title="Nc",
            range=[4.8, 7.65],
            dtick=0.5,
            gridcolor="#E9E0DB",
            zeroline=False,
            showline=True,
            linecolor="#CFC0B9",
        ),
        height=485,
        paper_bgcolor="#FFFDFC",
        plot_bgcolor="#FFFDFC",
        margin=dict(l=55, r=25, t=70, b=55),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.28,
            xanchor="left",
            x=0,
            font=dict(size=11, color="#6F625E"),
        ),
        hoverlabel=dict(bgcolor="#FFF8F3", font_color="#5C4B47"),
    )
    return fig


def half_excavation_animation_html(
    *,
    H: float,
    B: float,
    q: float,
    Su: float,
    gamma: float,
    animate: bool,
) -> str:
    """
    Half-model conceptual animation.

    The mesh deforms cyclically in the same visual spirit as a numerical
    deformation movie. The displayed displacement is intentionally exaggerated.
    """
    settings = {
        "H": H,
        "B": B,
        "q": q,
        "Su": Su,
        "gamma": gamma,
        "animate": animate,
    }
    settings_json = json.dumps(settings)

    return f"""
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <style>
        html, body {{ margin: 0; padding: 0; background: transparent; }}
        * {{ box-sizing: border-box; }}
        .card {{
          font-family: Inter, "Segoe UI", Arial, sans-serif;
          background: linear-gradient(145deg, #FFFDFC 0%, #FFF7F1 100%);
          border: 1px solid #E8DCD5;
          border-radius: 22px;
          padding: 14px 16px 12px;
          box-shadow: 0 12px 32px rgba(93, 75, 71, .08);
        }}
        .head {{
          display:flex; justify-content:space-between; align-items:flex-start;
          gap:12px; margin:2px 5px 9px;
        }}
        .title {{ color:#5C4B47; font-size:21px; font-weight:780; }}
        .subtitle {{ color:#8A746E; font-size:12.5px; margin-top:4px; line-height:1.4; }}
        .pill {{
          color:#5C4B47; background:#F4E6DE; border:1px solid #E6CEC3;
          padding:6px 10px; border-radius:999px; font-size:11.5px; font-weight:750;
          white-space:nowrap;
        }}
        canvas {{ width:100%; height:auto; display:block; border-radius:15px; }}
        .note {{ color:#8A746E; font-size:11.5px; line-height:1.45; margin:7px 7px 1px; }}
      </style>
    </head>
    <body>
      <div class="card">
        <div class="head">
          <div>
            <div class="title">Half excavation — animated basal-heave deformation</div>
            <div class="subtitle">Symmetry half-model • mesh motion is exaggerated to make the mechanism visible</div>
          </div>
          <div class="pill">{"ANIMATED" if animate else "PAUSED"}</div>
        </div>
        <canvas id="model" width="1120" height="650"></canvas>
        <div class="note">
          The curved failure surface follows the requested conceptual geometry: it starts at the excavation centreline,
          passes beneath the wall, rises toward an exterior shear plane, and continues vertically to ground level.
        </div>
      </div>

      <script>
        const S = {settings_json};
        const canvas = document.getElementById('model');
        const ctx = canvas.getContext('2d');
        const W = canvas.width, HH = canvas.height;

        const C = {{
          bg:'#FFFDFC', soil:'#F0E1D4', soil2:'#E6CFC0', mesh:'#93A5A7',
          meshSoft:'rgba(125,148,150,.60)', wall:'#52666A', strut:'#83A5A0',
          text:'#5C4B47', muted:'#8A746E', dim:'#6F8588', failure:'#C86E68',
          failFill:'rgba(216,132,124,.16)', surcharge:'#799C8E', excavation:'#FFFDFC',
          centre:'#B8AAA4', heave:'#D7998B', white:'#FFFDFC'
        }};

        const groundY = 105;
        const maxHalfWidth = 350;
        const maxDepth = 330;
        const scale = Math.min(maxHalfWidth / (S.B / 2), maxDepth / S.H);
        const halfWidth = (S.B / 2) * scale;
        const depth = S.H * scale;
        const centreX = 125;
        const wallX = centreX + halfWidth;
        const baseY = groundY + depth;
        const embed = Math.min(72, Math.max(34, depth * .19));
        const toeY = baseY + embed;
        const shearX = Math.min(1030, wallX + Math.max(135, depth * .55));
        const curveEndY = baseY + Math.min(28, depth * .10);
        const lowY = Math.min(590, baseY + Math.max(90, depth * .39));
        const xMax = 1080;
        const yMax = 610;

        const start = {{x: centreX, y: baseY}};
        const c1 = {{x: centreX + (wallX-centreX)*.45, y: baseY + 20}};
        const c2 = {{x: wallX + (shearX-wallX)*.12, y: lowY}};
        const end = {{x: shearX, y: curveEndY}};

        function bezier(t) {{
          const u=1-t;
          return {{
            x:u*u*u*start.x+3*u*u*t*c1.x+3*u*t*t*c2.x+t*t*t*end.x,
            y:u*u*u*start.y+3*u*u*t*c1.y+3*u*t*t*c2.y+t*t*t*end.y
          }};
        }}

        function roundedRect(x,y,w,h,r,fill,stroke=null) {{
          ctx.beginPath();
          ctx.roundRect(x,y,w,h,r);
          if(fill){{ctx.fillStyle=fill;ctx.fill();}}
          if(stroke){{ctx.strokeStyle=stroke;ctx.stroke();}}
        }}

        function arrow(x1,y1,x2,y2,color,width=2.5,head=8) {{
          const a=Math.atan2(y2-y1,x2-x1);
          ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=width;
          ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();
          ctx.beginPath();
          ctx.moveTo(x2,y2);
          ctx.lineTo(x2-head*Math.cos(a-Math.PI/6),y2-head*Math.sin(a-Math.PI/6));
          ctx.lineTo(x2-head*Math.cos(a+Math.PI/6),y2-head*Math.sin(a+Math.PI/6));
          ctx.closePath();ctx.fill();
        }}

        function dimension(x1,y1,x2,y2,label,vertical=false) {{
          ctx.save();
          ctx.strokeStyle=C.dim;ctx.fillStyle=C.dim;ctx.lineWidth=1.8;
          arrow(x1,y1,x2,y2,C.dim,1.8,7);arrow(x2,y2,x1,y1,C.dim,1.8,7);
          ctx.font='700 16px Segoe UI';
          if(vertical){{
            ctx.translate(x1-18,(y1+y2)/2);ctx.rotate(-Math.PI/2);
            const tw=ctx.measureText(label).width;
            roundedRect(-tw/2-8,-14,tw+16,28,14,'#FFF8F3','#E5D7D0');
            ctx.fillStyle=C.text;ctx.textAlign='center';ctx.fillText(label,0,6);
          }} else {{
            const mx=(x1+x2)/2,my=y1-15,tw=ctx.measureText(label).width;
            roundedRect(mx-tw/2-8,my-18,tw+16,28,14,'#FFF8F3','#E5D7D0');
            ctx.fillStyle=C.text;ctx.textAlign='center';ctx.fillText(label,mx,my+2);
          }}
          ctx.restore();
        }}

        // Structured triangular mesh. Nodes inside the excavation void are retained
        // in the grid but the associated triangles are not drawn.
        const dx=43, dy=38;
        const cols=Math.ceil((xMax-centreX)/dx)+1;
        const rows=Math.ceil((yMax-groundY)/dy)+1;
        const nodes=[];
        for(let j=0;j<rows;j++){{
          const row=[];
          for(let i=0;i<cols;i++){{
            const jitterX=((i*17+j*11)%9-4)*1.7;
            const jitterY=((i*7+j*19)%9-4)*1.3;
            row.push({{x:centreX+i*dx+jitterX,y:groundY+j*dy+jitterY}});
          }}
          nodes.push(row);
        }}

        function inExcavation(x,y) {{
          return x < wallX-3 && y > groundY-2 && y < baseY-1;
        }}

        function displacement(p,x,y) {{
          // Upward basal heave below the excavation.
          const bx=(x-(centreX+halfWidth*.55))/(Math.max(halfWidth*.75,80));
          const by=(y-(baseY+55))/(Math.max(depth*.50,110));
          const basal=Math.exp(-(bx*bx+by*by));

          // Downward/inward retained block between the wall and vertical shear plane.
          const rx=(x-(wallX+(shearX-wallX)*.42))/Math.max((shearX-wallX)*.65,90);
          const ry=(y-(groundY+depth*.42))/Math.max(depth*.70,130);
          const retained=Math.exp(-(rx*rx+ry*ry));
          const retainedMask=(x>=wallX && x<=shearX && y<=curveEndY+25)?1:0;

          // Localized distortion around the curved slip zone.
          let minD=9999;
          for(let k=0;k<=24;k++){{
            const b=bezier(k/24);const dd=Math.hypot(x-b.x,y-b.y);if(dd<minD)minD=dd;
          }}
          const slip=Math.exp(-(minD*minD)/(2*68*68));

          let ux=0,uy=0;
          ux += -5.0*basal*p;
          uy += -17.0*basal*p;
          ux += -13.0*retained*retainedMask*p;
          uy +=  10.0*retained*retainedMask*p;
          ux += -5.5*slip*p;
          uy += -2.0*slip*p;

          // Symmetry boundary has zero horizontal movement.
          if(Math.abs(x-centreX)<18) ux=0;
          // Far field restraint.
          const far=Math.max(0,Math.min(1,(xMax-x)/145));
          ux*=far;uy*=far;
          return {{x:x+ux,y:y+uy}};
        }}

        function drawTriangle(a,b,c,p) {{
          const cx=(a.x+b.x+c.x)/3,cy=(a.y+b.y+c.y)/3;
          if(inExcavation(cx,cy)) return;
          const A=displacement(p,a.x,a.y),B=displacement(p,b.x,b.y),D=displacement(p,c.x,c.y);
          ctx.beginPath();ctx.moveTo(A.x,A.y);ctx.lineTo(B.x,B.y);ctx.lineTo(D.x,D.y);ctx.closePath();
          ctx.strokeStyle=C.meshSoft;ctx.lineWidth=1.15;ctx.stroke();
        }}

        function drawFailureSurface(dashOffset) {{
          ctx.save();
          ctx.beginPath();ctx.moveTo(start.x,start.y);
          ctx.bezierCurveTo(c1.x,c1.y,c2.x,c2.y,end.x,end.y);
          ctx.lineTo(shearX,groundY);
          ctx.lineWidth=4.2;ctx.strokeStyle=C.failure;ctx.setLineDash([13,9]);
          ctx.lineDashOffset=-dashOffset;ctx.stroke();ctx.restore();
        }}

        function drawFrame(time) {{
          const pulse=S.animate ? (1-Math.cos(time/1000*Math.PI))/2 : .72;
          const dash=S.animate ? time/25 : 0;

          ctx.clearRect(0,0,W,HH);ctx.fillStyle=C.bg;ctx.fillRect(0,0,W,HH);

          // Soil mass, then excavation void.
          ctx.fillStyle=C.soil;ctx.fillRect(centreX,groundY,xMax-centreX,yMax-groundY);
          ctx.fillStyle=C.excavation;ctx.fillRect(centreX,groundY,wallX-centreX,baseY-groundY);

          // Soft failure-zone fill.
          ctx.save();ctx.beginPath();ctx.moveTo(start.x,start.y);
          ctx.bezierCurveTo(c1.x,c1.y,c2.x,c2.y,end.x,end.y);
          ctx.lineTo(shearX,groundY);ctx.lineTo(wallX,groundY);ctx.lineTo(wallX,baseY);ctx.lineTo(start.x,start.y);
          ctx.fillStyle=C.failFill;ctx.fill();ctx.restore();

          // Mesh.
          for(let j=0;j<rows-1;j++){{
            for(let i=0;i<cols-1;i++){{
              const a=nodes[j][i],b=nodes[j][i+1],c=nodes[j+1][i],d=nodes[j+1][i+1];
              if((i+j)%2===0){{drawTriangle(a,b,d,pulse);drawTriangle(a,d,c,pulse);}}
              else{{drawTriangle(a,b,c,pulse);drawTriangle(b,d,c,pulse);}}
            }}
          }}

          // Ground line and symmetry centreline.
          ctx.strokeStyle=C.dim;ctx.lineWidth=2.4;ctx.beginPath();ctx.moveTo(centreX,groundY);ctx.lineTo(xMax,groundY);ctx.stroke();
          ctx.save();ctx.strokeStyle=C.centre;ctx.setLineDash([8,7]);ctx.lineWidth=2;
          ctx.beginPath();ctx.moveTo(centreX,55);ctx.lineTo(centreX,yMax);ctx.stroke();ctx.restore();
          ctx.fillStyle=C.muted;ctx.font='600 13px Segoe UI';ctx.fillText('C/L — symmetry',centreX+8,73);

          // Animated heaving excavation base.
          ctx.strokeStyle=C.heave;ctx.lineWidth=4;ctx.beginPath();
          const segments=28;
          for(let i=0;i<=segments;i++){{
            const x=centreX+(wallX-centreX)*i/segments;
            const shape=Math.sin(Math.PI*i/segments);
            const y=baseY-11*pulse*shape;
            if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);
          }}ctx.stroke();

          // Retaining wall: top fixed, toe moves slightly toward excavation.
          const toeShift=-7*pulse;
          ctx.strokeStyle=C.wall;ctx.lineWidth=9;ctx.lineCap='round';ctx.beginPath();
          ctx.moveTo(wallX,groundY);ctx.quadraticCurveTo(wallX-1*pulse,baseY,wallX+toeShift,toeY);ctx.stroke();

          // Half struts ending at the symmetry line.
          const levels=[.24,.51,.78];
          ctx.strokeStyle=C.strut;ctx.lineWidth=7;ctx.lineCap='round';
          for(const f of levels){{
            const y=groundY+depth*f;
            const wallAt=wallX-4*pulse*f*f;
            ctx.beginPath();ctx.moveTo(centreX+7,y);ctx.lineTo(wallAt-5,y);ctx.stroke();
            ctx.fillStyle=C.wall;ctx.beginPath();ctx.arc(wallAt-5,y,5,0,Math.PI*2);ctx.fill();
          }}

          // Failure surface and motion indicators.
          drawFailureSurface(dash);
          arrow(centreX+halfWidth*.55,baseY+34,centreX+halfWidth*.55,baseY-39-8*pulse,C.failure,3.3,10);
          arrow(wallX+(shearX-wallX)*.45,groundY+45,wallX+(shearX-wallX)*.40,groundY+78+8*pulse,C.failure,2.6,9);
          arrow(wallX+(shearX-wallX)*.23,groundY+depth*.50,wallX+(shearX-wallX)*.14,groundY+depth*.54,C.failure,2.6,9);

          // Uniform surcharge on retained ground only.
          for(let x=wallX+45;x<Math.min(xMax-35,wallX+300);x+=55){{
            arrow(x,45,x,89,C.surcharge,2.4,8);
          }}
          ctx.fillStyle=C.text;ctx.font='700 15px Segoe UI';ctx.fillText(`q = ${{S.q.toFixed(1)}} kPa`,Math.min(xMax-150,wallX+55),35);

          // Dimensions. The model is half the excavation; full B remains a given input.
          ctx.strokeStyle=C.dim;ctx.lineWidth=1.3;
          ctx.beginPath();ctx.moveTo(centreX,groundY-8);ctx.lineTo(centreX,18);ctx.moveTo(wallX,groundY-8);ctx.lineTo(wallX,18);ctx.stroke();
          dimension(centreX,28,wallX,28,`B/2 = ${{(S.B/2).toFixed(2)}} m`);
          ctx.beginPath();ctx.moveTo(centreX-10,groundY);ctx.lineTo(52,groundY);ctx.moveTo(centreX-10,baseY);ctx.lineTo(52,baseY);ctx.stroke();
          dimension(63,groundY,63,baseY,`H = ${{S.H.toFixed(2)}} m`,true);

          // Input badge.
          roundedRect(735,548,335,66,15,'rgba(255,249,245,.94)','#E4D6CE');
          ctx.fillStyle=C.text;ctx.font='750 14px Segoe UI';ctx.fillText(`Given full width B = ${{S.B.toFixed(2)}} m`,755,573);
          ctx.fillText(`Su = ${{S.Su.toFixed(1)}} kPa   γ = ${{S.gamma.toFixed(1)}} kN/m³`,755,597);

          ctx.fillStyle=C.muted;ctx.font='600 12px Segoe UI';
          ctx.fillText('Conceptual deformed mesh — displacement magnified',138,635);

          if(S.animate) requestAnimationFrame(drawFrame);
        }}

        requestAnimationFrame(drawFrame);
      </script>
    </body>
    </html>
    """


st.set_page_config(
    page_title="Basal Heave Stability Calculator",
    page_icon="🏗️",
    layout="wide",
)

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
      }
      .stApp {
        background:
          radial-gradient(circle at 10% 5%, rgba(240,219,207,.27), transparent 28%),
          linear-gradient(180deg,#FFFDFC 0%,var(--page) 100%);
        color:var(--text);
      }
      .block-container { max-width:1500px; padding-top:1.45rem; padding-bottom:3rem; }
      h1,h2,h3,h4,p,label { color:var(--text); }
      [data-testid="stCaptionContainer"] { color:var(--muted); }
      [data-testid="stMetric"] {
        background:linear-gradient(145deg,#FFFDFC 0%,#FFF7F2 100%);
        border:1px solid var(--line); border-radius:18px; padding:16px 18px;
        box-shadow:0 8px 24px rgba(93,75,71,.06);
      }
      [data-testid="stMetricLabel"] { color:var(--muted); }
      [data-testid="stExpander"] {
        background:rgba(255,253,252,.80); border:1px solid var(--line);
        border-radius:16px; overflow:hidden;
      }
      [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:16px; overflow:hidden; }
      div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {
        background:#FFFDFC; border-color:#DDC9BF; border-radius:12px;
      }
      .stDownloadButton > button {
        border:1px solid #D4B8AB; border-radius:12px;
        background:linear-gradient(145deg,#C99A8A,#B98778); color:white; font-weight:700;
      }
      [data-testid="stAlert"] { border-radius:14px; }
      hr { border-color:var(--line); }
      .scope-box {
        background:linear-gradient(145deg,#FFF9F5,#FFFDFC); border:1px solid var(--line);
        border-left:5px solid var(--accent); border-radius:14px; padding:.9rem 1.1rem;
        margin:.5rem 0 1rem; color:var(--text);
      }
      .section-note {
        background:#FFF8F4; border:1px solid var(--line); border-radius:14px;
        color:var(--muted); padding:12px 15px; line-height:1.55;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

engine = get_engine()

st.title("Basal Heave Stability Calculator")
st.caption(
    "Terzaghi and Bjerrum & Eide are calculated analytically • "
    "only the Optum bounds are interpolated from the numerical database"
)

with st.expander("Calculation basis and database scope", expanded=False):
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
        "The two classical factors of safety and Nc are recalculated for every input."
    )
    st.write(
        "The only absent Optum corner is q = 20 kPa with Su = 5 kPa. "
        "For 10 < q ≤ 20 kPa, the supported boundary is Su ≥ q/2."
    )

st.subheader("Given input parameters")

input_cols = st.columns(5)
with input_cols[2]:
    q = st.number_input(
        "Uniform surcharge, q (kPa)", min_value=0.0, max_value=20.0,
        value=10.0, step=1.0,
    )

minimum_su = engine.minimum_su_for_q(q)
with input_cols[0]:
    Su = st.number_input(
        "Undrained shear strength, Su (kPa)",
        min_value=float(minimum_su), max_value=60.0,
        value=max(20.0, float(minimum_su)), step=1.0,
        help=f"The Optum database requires Su ≥ {minimum_su:g} kPa when q = {q:g} kPa.",
    )
with input_cols[1]:
    gamma = st.number_input(
        "Total unit weight, γ (kN/m³)", min_value=16.0, max_value=20.0,
        value=18.0, step=0.5,
    )
with input_cols[3]:
    H = st.number_input(
        "Excavation depth, H (m)", min_value=6.0, max_value=14.0,
        value=10.0, step=0.5,
    )
with input_cols[4]:
    B = st.number_input(
        "Full excavation width, B (m)", min_value=4.0, max_value=30.0,
        value=10.0, step=0.5,
    )

settings_cols = st.columns([1.45, 1.0, 1.0])
with settings_cols[0]:
    recommendation_basis = st.selectbox(
        "Recommended Optum value", options=["lower", "average", "upper"],
        format_func=basis_label, index=1,
    )
with settings_cols[1]:
    target_fs = st.number_input("Target F.S.", min_value=0.0, value=1.50, step=0.05)
with settings_cols[2]:
    animate_mechanism = st.toggle("Animate deformation", value=True)

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

main_results = st.columns(3)
main_results[0].metric("Terzaghi F.S. — equation", format_fs(result["terzaghi"]))
main_results[1].metric("Bjerrum & Eide F.S. — equation", f"{result['bjerrum']:.3f}")
main_results[2].metric("Recommended Optum F.S. — interpolation", f"{result['recommended']:.3f}")

if result["terzaghi"] is None:
    st.warning(
        "The Terzaghi denominator is zero or negative for this input "
        f"({result['terzaghi_denominator']:.3f} kPa), so a positive Terzaghi F.S. is not reported."
    )

optum_results = st.columns(4)
optum_results[0].metric("Optum lower bound", f"{result['numerical_lower']:.3f}")
optum_results[1].metric("Optum average", f"{result['numerical_average']:.3f}")
optum_results[2].metric("Optum upper bound", f"{result['numerical_upper']:.3f}")
optum_results[3].metric("Upper–lower gap", f"{result['bound_gap_percent']:.1f}%")

difference = result["recommended"] - target_fs
if difference >= 0:
    st.success(f"The recommended Optum F.S. is {difference:.3f} above the target of {target_fs:.2f}.")
else:
    st.error(f"The recommended Optum F.S. is {abs(difference):.3f} below the target of {target_fs:.2f}.")

# No equations are displayed externally. Nc is communicated through the graph.
visual_col, nc_col = st.columns([1.65, 1.0], gap="large")
with visual_col:
    st.subheader("Half excavation and animated failure mechanism")
    components.html(
        half_excavation_animation_html(
            H=H, B=B, q=q, Su=Su, gamma=gamma, animate=animate_mechanism,
        ),
        height=725,
        scrolling=False,
    )
with nc_col:
    st.subheader("Nc selection")
    st.plotly_chart(
        nc_chart(result["H_over_B"], result["Nc"]),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    st.metric("Calculated Nc", f"{result['Nc']:.3f}")
    st.caption(
        f"Selected geometry: H/B = {result['H_over_B']:.3f}. "
        "The graph is generated from the same piecewise rule used in the calculation."
    )

calculation_details = pd.DataFrame(
    [
        ["H/B", result["H_over_B"], "—"],
        ["B/H", result["B_over_H"], "—"],
        ["Half-model width", B / 2.0, "m"],
        ["B₁", result["B1"], "m"],
        ["γH + q", result["driving_pressure"], "kPa"],
        ["Terzaghi denominator", result["terzaghi_denominator"], "kPa"],
        ["Bjerrum Nc", result["Nc"], "—"],
        ["Optum recommendation basis", basis_label(recommendation_basis), "—"],
    ],
    columns=["Quantity", "Value", "Unit"],
)
st.dataframe(calculation_details, use_container_width=True, hide_index=True)

st.markdown(
    """
    <div class="section-note">
      <b>Visual interpretation:</b> only one half of the excavation is shown because the section is symmetric.
      The animated mesh represents an exaggerated deformation pattern, not a calculated Optum displacement field.
      The dashed curve is a conceptual basal-heave surface matching the requested geometry.
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader("Nearest original Optum models")
nearest = pd.DataFrame(engine.nearest_cases(Su=Su, gamma=gamma, q=q, H=H, B=B, count=8))
st.dataframe(nearest, use_container_width=True, hide_index=True)

export_rows = [
    ("Su (kPa)", result["Su"]),
    ("gamma (kN/m3)", result["gamma"]),
    ("q (kPa)", result["q"]),
    ("H (m)", result["H"]),
    ("Full B (m)", result["B"]),
    ("Half-model width B/2 (m)", result["B"] / 2.0),
    ("H/B", result["H_over_B"]),
    ("B/H", result["B_over_H"]),
    ("B1 (m)", result["B1"]),
    ("Terzaghi F.S. calculated analytically", result["terzaghi"]),
    ("Terzaghi denominator (kPa)", result["terzaghi_denominator"]),
    ("Bjerrum Nc calculated analytically", result["Nc"]),
    ("Bjerrum & Eide F.S. calculated analytically", result["bjerrum"]),
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
- Homogeneous clay with constant undrained shear strength.
- Undrained total-stress condition.
- The wall, supports, groundwater, boundaries, and hard-stratum condition must match the numerical study.
- Optum interpolation is blocked outside the data-supported domain.
- The target F.S. is user-defined and is not presented as a universal code requirement.
"""
)
