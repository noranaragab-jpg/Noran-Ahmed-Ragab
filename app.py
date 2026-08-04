from __future__ import annotations

from pathlib import Path
import csv
import io
import math
from string import Template

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


def build_nc_chart(*, h_over_b: float, nc: float) -> str:
    """Draw the Bjerrum & Eide Nc relationship directly from the equations."""
    width = 980
    height = 560
    left = 86
    right = 34
    top = 42
    bottom = 155
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_min, x_max = 0.0, 6.0
    y_min, y_max = 5.0, 7.6

    def nc_equation(x: float) -> float:
        if x > 3.0:
            return 7.2
        return 5.14 + 1.05 * x - 0.18 * x**2

    def sx(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y: float) -> float:
        return top + (y_max - y) / (y_max - y_min) * plot_h

    polynomial_points = []
    for i in range(121):
        x = 3.0 * i / 120.0
        polynomial_points.append(f"{sx(x):.2f},{sy(nc_equation(x)):.2f}")

    plateau_points = [
        f"{sx(3.0):.2f},{sy(7.2):.2f}",
        f"{sx(6.0):.2f},{sy(7.2):.2f}",
    ]

    x_selected = min(max(h_over_b, x_min), x_max)
    y_selected = nc

    vertical_grid = "".join(
        f'<line class="grid" x1="{sx(x):.2f}" y1="{top}" '
        f'x2="{sx(x):.2f}" y2="{top + plot_h}" />'
        f'<text class="tick" x="{sx(x):.2f}" y="{top + plot_h + 30}" '
        f'text-anchor="middle">{x:g}</text>'
        for x in range(0, 7)
    )
    y_ticks = [5.0, 5.5, 6.0, 6.5, 7.0, 7.5]
    horizontal_grid = "".join(
        f'<line class="grid" x1="{left}" y1="{sy(y):.2f}" '
        f'x2="{left + plot_w}" y2="{sy(y):.2f}" />'
        f'<text class="tick" x="{left - 17}" y="{sy(y) + 5:.2f}" '
        f'text-anchor="end">{y:.1f}</text>'
        for y in y_ticks
    )

    template = Template(r'''
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  html, body { margin: 0; padding: 0; background: transparent; font-family: "Segoe UI", Arial, sans-serif; }
  .card {
    background: linear-gradient(180deg, #fffdfa 0%, #faf7f1 100%);
    border: 1px solid #e4ddd2;
    border-radius: 18px;
    padding: 12px 14px 8px;
    box-shadow: 0 10px 28px rgba(73, 86, 89, 0.07);
  }
  svg { width: 100%; height: auto; display: block; }
  .grid { stroke: #dedbd4; stroke-width: 1; }
  .axis { stroke: #536b72; stroke-width: 2.2; }
  .curve { fill: none; stroke: #789b95; stroke-width: 4; stroke-linecap: round; stroke-linejoin: round; }
  .transition { stroke: #789b95; stroke-width: 2.2; stroke-dasharray: 6 5; }
  .selected { stroke: #c98784; stroke-width: 2; stroke-dasharray: 7 6; }
  .selected-dot { fill: #c98784; stroke: #fffdfa; stroke-width: 4; }
  .tick { fill: #687b80; font-size: 14px; }
  .axis-label { fill: #40555d; font-size: 17px; font-weight: 750; }
  .title { fill: #34464f; font-size: 20px; font-weight: 780; }
  .subtitle { fill: #718287; font-size: 13px; }
  .formula-box { fill: #f3eee6; stroke: #ddd2c3; }
  .formula-title { fill: #6d7e82; font-size: 12px; font-weight: 750; letter-spacing: 0.5px; }
  .formula { fill: #3f565d; font-size: 14px; font-weight: 650; }
  .value-box { fill: #eef3f0; stroke: #cddcd6; }
  .value-label { fill: #6b7d81; font-size: 12px; font-weight: 700; }
  .value { fill: #36524f; font-size: 19px; font-weight: 800; }
</style>
</head>
<body>
<div class="card">
<svg viewBox="0 0 $WIDTH $HEIGHT" role="img" aria-label="Calculated Nc versus H over B">
  <text class="title" x="$LEFT" y="25">Bjerrum &amp; Eide bearing-capacity factor, N<tspan baseline-shift="sub" font-size="14">c</tspan></text>
  <text class="subtitle" x="$LEFT" y="45">Equation-generated curve for an infinitely long excavation (B/L ≈ 0)</text>

  $VGRID
  $HGRID

  <line class="axis" x1="$LEFT" y1="$TOP" x2="$LEFT" y2="$PLOT_BOTTOM" />
  <line class="axis" x1="$LEFT" y1="$PLOT_BOTTOM" x2="$PLOT_RIGHT" y2="$PLOT_BOTTOM" />

  <polyline class="curve" points="$POLY_POINTS" />
  <line class="transition" x1="$X3" y1="$Y_POLY3" x2="$X3" y2="$Y72" />
  <polyline class="curve" points="$PLATEAU_POINTS" />

  <line class="selected" x1="$XSEL" y1="$TOP" x2="$XSEL" y2="$PLOT_BOTTOM" />
  <line class="selected" x1="$LEFT" y1="$YSEL" x2="$XSEL" y2="$YSEL" />
  <circle class="selected-dot" cx="$XSEL" cy="$YSEL" r="8" />

  <text class="axis-label" x="$XCENTER" y="$AXIS_LABEL_Y" text-anchor="middle">H/B</text>
  <text class="axis-label" transform="translate(26,$YCENTER) rotate(-90)" text-anchor="middle">N<tspan baseline-shift="sub" font-size="12">c</tspan></text>

  <rect class="value-box" x="$VALUE_X" y="61" width="190" height="70" rx="12" />
  <text class="value-label" x="$VALUE_TEXT_X" y="85">CURRENT INPUT</text>
  <text class="value" x="$VALUE_TEXT_X" y="113">H/B = $HB_VALUE   •   N<tspan baseline-shift="sub" font-size="13">c</tspan> = $NC_VALUE</text>

  <rect class="formula-box" x="$LEFT" y="$FORMULA_Y" width="650" height="54" rx="11" />
  <text class="formula-title" x="$FORMULA_TEXT_X" y="$FORMULA_TITLE_Y">EQUATIONS USED IN THE APPLICATION</text>
  <text class="formula" x="$FORMULA_TEXT_X" y="$FORMULA_LINE_Y">N<tspan baseline-shift="sub" font-size="11">c</tspan> = 5.14 + 1.05(H/B) − 0.18(H/B)² for H/B ≤ 3;  N<tspan baseline-shift="sub" font-size="11">c</tspan> = 7.20 for H/B &gt; 3</text>
</svg>
</div>
</body>
</html>
''')

    return template.substitute(
        WIDTH=width,
        HEIGHT=height,
        LEFT=left,
        TOP=top,
        PLOT_BOTTOM=top + plot_h,
        PLOT_RIGHT=left + plot_w,
        VGRID=vertical_grid,
        HGRID=horizontal_grid,
        POLY_POINTS=" ".join(polynomial_points),
        PLATEAU_POINTS=" ".join(plateau_points),
        X3=f"{sx(3.0):.2f}",
        Y_POLY3=f"{sy(nc_equation(3.0)):.2f}",
        Y72=f"{sy(7.2):.2f}",
        XSEL=f"{sx(x_selected):.2f}",
        YSEL=f"{sy(y_selected):.2f}",
        XCENTER=f"{left + plot_w / 2:.2f}",
        YCENTER=f"{top + plot_h / 2:.2f}",
        AXIS_LABEL_Y=top + plot_h + 58,
        VALUE_X=width - 234,
        VALUE_TEXT_X=width - 216,
        HB_VALUE=f"{h_over_b:.3f}",
        NC_VALUE=f"{nc:.3f}",
        FORMULA_Y=height - 73,
        FORMULA_TEXT_X=left + 18,
        FORMULA_TITLE_Y=height - 51,
        FORMULA_LINE_Y=height - 29,
    )


def build_excavation_animation(
    *,
    H: float,
    B: float,
    q: float,
    Su: float,
    gamma: float,
    animate: bool,
) -> str:
    """Create an Optum-style deforming triangular mesh for basal heave."""
    visual_ratio = B / H
    excavation_width = max(220.0, min(500.0, 285.0 * visual_ratio))
    b1 = B / math.sqrt(2.0)

    template = Template(r'''
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  html, body { margin: 0; padding: 0; background: transparent; font-family: "Segoe UI", Arial, sans-serif; }
  .card {
    background: linear-gradient(180deg, #fffdfa 0%, #faf7f1 100%);
    border: 1px solid #e4ddd2;
    border-radius: 18px;
    padding: 10px 12px 8px;
    box-shadow: 0 10px 28px rgba(73, 86, 89, 0.08);
  }
  canvas { width: 100%; height: auto; display: block; border-radius: 13px; }
  .caption {
    display: flex;
    justify-content: space-between;
    gap: 14px;
    align-items: center;
    color: #6c7f84;
    font-size: 12px;
    padding: 5px 8px 1px;
  }
  .legend { display: flex; gap: 16px; flex-wrap: wrap; }
  .key { display: inline-flex; align-items: center; gap: 6px; }
  .line { width: 28px; height: 0; border-top: 3px solid #789b95; }
  .dash { width: 28px; height: 0; border-top: 3px dashed #c98784; }
</style>
</head>
<body>
<div class="card">
  <canvas id="meshCanvas" aria-label="Animated triangular mesh showing a conceptual basal-heave mechanism"></canvas>
  <div class="caption">
    <div class="legend">
      <span class="key"><span class="line"></span>deformed soil mesh</span>
      <span class="key"><span class="dash"></span>conceptual failure envelope</span>
    </div>
    <span>Motion is illustrative, not a calculated displacement field.</span>
  </div>
</div>
<script>
(() => {
  const canvas = document.getElementById('meshCanvas');
  const ctx = canvas.getContext('2d');
  const W = 1160;
  const Hc = 640;
  const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
  canvas.width = W * dpr;
  canvas.height = Hc * dpr;
  canvas.style.aspectRatio = W + ' / ' + Hc;
  ctx.scale(dpr, dpr);

  const ground = 112;
  const base = 365;
  const cx = 600;
  const excW = $EXC_W;
  const halfB = excW / 2;
  const left = cx - halfB;
  const right = cx + halfB;
  const wallToe = base + 45;
  const animate = $ANIMATE;

  const palette = {
    bg: '#fffdfa', soil: '#eee4d5', soilDeep: '#e6d6c1',
    mesh: 'rgba(87, 105, 110, 0.48)', meshStrong: 'rgba(84, 105, 109, 0.72)',
    wall: '#6c8790', strut: '#8ca9a4', ground: '#4f686f',
    rose: '#c98784', sage: '#789b95', text: '#3f555c',
    muted: '#718287', dim: '#607981', load: '#7f9f8f'
  };

  function gauss(v, s) { return Math.exp(-(v * v) / (2 * s * s)); }
  function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }

  const cols = 37;
  const xCoords = [];
  for (let i = 0; i < cols; i++) {
    const u = 2 * i / (cols - 1) - 1;
    const mapped = Math.sign(u) * Math.pow(Math.abs(u), 1.55);
    xCoords.push(cx + mapped * 555);
  }
  const yCoords = [ground, 132, 157, 188, 222, 258, 297, 333, base, 397, 432, 473, 520, 570, 620];

  function jitter(i, j, scale) { return Math.sin(i * 12.9898 + j * 78.233) * scale; }

  const nodes = yCoords.map((y, j) => xCoords.map((x, i) => ({
    x: x + (i > 0 && i < cols - 1 ? jitter(i, j, 4.2) : 0),
    y: y + (j > 0 && j < yCoords.length - 1 ? jitter(j, i, 2.7) : 0)
  })));

  const triangles = [];
  for (let j = 0; j < nodes.length - 1; j++) {
    for (let i = 0; i < cols - 1; i++) {
      const a = nodes[j][i]; const b = nodes[j][i + 1];
      const c = nodes[j + 1][i]; const d = nodes[j + 1][i + 1];
      if ((i + j) % 2 === 0) triangles.push([a, b, d], [a, d, c]);
      else triangles.push([a, b, c], [b, d, c]);
    }
  }

  function isInExcavation(x, y) { return x > left && x < right && y >= ground - 2 && y < base; }

  function displacement(p, phase) {
    const x = p.x; const y = p.y;
    const z = Math.max(0, y - ground);
    const belowBase = Math.max(0, y - base);
    const absCenter = Math.abs(x - cx);
    const bulbWidth = halfB * 1.08 + 0.42 * belowBase + 70;
    const upliftCore = gauss(x - cx, bulbWidth) * gauss(belowBase, 185);
    const baseInfluence = y >= base - 45 ? 1.0 : clamp((y - (base - 125)) / 80, 0, 1);
    const uplift = -34 * upliftCore * baseInfluence;
    const wallDistance = Math.min(Math.abs(x - left), Math.abs(x - right));
    const sideZone = gauss(wallDistance, halfB * 0.47 + 74) * gauss(z, 235);
    const outsideMask = absCenter >= halfB * 0.90 ? 1 : 0.25;
    const settlement = 18 * sideZone * outsideMask;
    const inward = Math.sign(cx - x) * 15 * sideZone * outsideMask * (0.45 + 0.55 * clamp(z / 320, 0, 1));
    const deepSpread = Math.sign(x - cx) * 5.5 * upliftCore * clamp(belowBase / 230, 0, 1);
    return { x: x + phase * (inward + deepSpread), y: y + phase * (uplift + settlement) };
  }

  function groundSettlement(x, phase) {
    const nearest = Math.min(Math.abs(x - left), Math.abs(x - right));
    const outside = (x <= left || x >= right) ? 1 : 0;
    return phase * outside * 17 * gauss(nearest, halfB * 0.48 + 80);
  }
  function baseHeave(x, phase) { return -phase * 31 * gauss(x - cx, halfB * 0.82); }

  function drawArrow(x1, y1, x2, y2, color, width = 2.3) {
    const ang = Math.atan2(y2 - y1, x2 - x1);
    ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = width;
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - 9 * Math.cos(ang - Math.PI / 6), y2 - 9 * Math.sin(ang - Math.PI / 6));
    ctx.lineTo(x2 - 9 * Math.cos(ang + Math.PI / 6), y2 - 9 * Math.sin(ang + Math.PI / 6));
    ctx.closePath(); ctx.fill();
  }

  function drawDimension(x1, y1, x2, y2, label, rotate = false) {
    ctx.save(); ctx.strokeStyle = palette.dim; ctx.fillStyle = palette.dim; ctx.lineWidth = 1.7;
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    drawArrow(x1 + (x2 - x1) * 0.48, y1 + (y2 - y1) * 0.48, x1, y1, palette.dim, 1.7);
    drawArrow(x1 + (x2 - x1) * 0.52, y1 + (y2 - y1) * 0.52, x2, y2, palette.dim, 1.7);
    ctx.font = '700 16px Segoe UI, Arial';
    if (!rotate) { ctx.textAlign = 'center'; ctx.fillText(label, (x1 + x2) / 2, y1 - 8); }
    else { ctx.translate(x1 + 24, (y1 + y2) / 2); ctx.rotate(Math.PI / 2); ctx.textAlign = 'center'; ctx.fillText(label, 0, 0); }
    ctx.restore();
  }

  function drawSurface(phase) {
    ctx.strokeStyle = palette.ground; ctx.lineWidth = 3;
    ctx.beginPath();
    for (let x = 28; x <= left; x += 8) { const y = ground + groundSettlement(x, phase); if (x === 28) ctx.moveTo(x, y); else ctx.lineTo(x, y); }
    ctx.stroke();
    ctx.beginPath();
    for (let x = right; x <= W - 28; x += 8) { const y = ground + groundSettlement(x, phase); if (x === right) ctx.moveTo(x, y); else ctx.lineTo(x, y); }
    ctx.stroke();
    ctx.strokeStyle = palette.sage; ctx.lineWidth = 4; ctx.beginPath();
    for (let x = left + 7; x <= right - 7; x += 8) { const y = base + baseHeave(x, phase); if (x <= left + 8) ctx.moveTo(x, y); else ctx.lineTo(x, y); }
    ctx.stroke();
  }

  function drawFailureEnvelope(phase, time) {
    const depth = Math.min(225, halfB * 0.82 + 95);
    const leftStartX = left - halfB * 0.62;
    const rightStartX = right + halfB * 0.62;
    const leftStartY = ground + groundSettlement(leftStartX, phase);
    const rightStartY = ground + groundSettlement(rightStartX, phase);
    const bottomY = base + depth - phase * 5;
    ctx.save(); ctx.lineCap = 'round'; ctx.setLineDash([12, 9]); ctx.lineDashOffset = -time * 0.045;
    ctx.strokeStyle = palette.rose; ctx.lineWidth = 4; ctx.beginPath();
    ctx.moveTo(leftStartX, leftStartY);
    ctx.bezierCurveTo(left - halfB * 0.34, base + depth * 0.36, cx - halfB * 0.42, bottomY, cx, bottomY);
    ctx.bezierCurveTo(cx + halfB * 0.42, bottomY, right + halfB * 0.34, base + depth * 0.36, rightStartX, rightStartY);
    ctx.stroke(); ctx.setLineDash([]); ctx.strokeStyle = 'rgba(201, 135, 132, 0.14)'; ctx.lineWidth = 18; ctx.stroke(); ctx.restore();
  }

  function drawMesh(phase) {
    ctx.lineWidth = 1.0;
    for (const tri of triangles) {
      const centroid = { x: (tri[0].x + tri[1].x + tri[2].x) / 3, y: (tri[0].y + tri[1].y + tri[2].y) / 3 };
      if (isInExcavation(centroid.x, centroid.y)) continue;
      const p0 = displacement(tri[0], phase); const p1 = displacement(tri[1], phase); const p2 = displacement(tri[2], phase);
      const under = centroid.y >= base - 10 && Math.abs(centroid.x - cx) < halfB * 1.55 + 180;
      ctx.strokeStyle = under ? palette.meshStrong : palette.mesh;
      ctx.beginPath(); ctx.moveTo(p0.x, p0.y); ctx.lineTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.closePath(); ctx.stroke();
    }
  }

  function drawStructure(phase) {
    ctx.fillStyle = palette.bg; ctx.fillRect(left + 5, ground - 4, excW - 10, base - ground + 8);
    ctx.strokeStyle = palette.wall; ctx.lineWidth = 9; ctx.lineCap = 'round'; ctx.beginPath();
    ctx.moveTo(left, ground); ctx.lineTo(left, wallToe); ctx.moveTo(right, ground); ctx.lineTo(right, wallToe); ctx.stroke();
    ctx.strokeStyle = palette.strut; ctx.lineWidth = 7;
    const strutYs = [ground + 64, ground + 130, ground + 198];
    for (const y of strutYs) { ctx.beginPath(); ctx.moveTo(left + 7, y); ctx.lineTo(right - 7, y); ctx.stroke(); }
    ctx.strokeStyle = palette.sage; ctx.lineWidth = 4; ctx.beginPath();
    for (let x = left + 7; x <= right - 7; x += 7) { const y = base + baseHeave(x, phase); if (x <= left + 8) ctx.moveTo(x, y); else ctx.lineTo(x, y); }
    ctx.stroke();
  }

  function drawLoads() {
    ctx.fillStyle = palette.text; ctx.font = '700 16px Segoe UI, Arial'; ctx.textAlign = 'left'; ctx.fillText('q = $Q kPa', right + 112, 66);
    const xs = [left - 125, left - 85, left - 45, right + 45, right + 85, right + 125];
    for (const x of xs) drawArrow(x, 42, x, 91, palette.load, 2.2);
  }

  function drawLabels(phase) {
    ctx.fillStyle = palette.text; ctx.font = '800 18px Segoe UI, Arial'; ctx.textAlign = 'center';
    ctx.fillText('BRACED EXCAVATION — DEFORMING MESH', cx, ground + 28);
    ctx.font = '650 13px Segoe UI, Arial'; ctx.fillStyle = palette.muted;
    ctx.fillText('soil settles beside the excavation while the base moves upward', cx, ground + 50);
    drawDimension(left, 28, right, 28, 'B = $B m');
    ctx.strokeStyle = '#98a8ab'; ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(left, 32); ctx.lineTo(left, ground - 7); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(right, 32); ctx.lineTo(right, ground - 7); ctx.stroke();
    drawDimension(right + 68, ground, right + 68, base, 'H = $H m', true);
    ctx.beginPath(); ctx.moveTo(right + 7, ground); ctx.lineTo(right + 70, ground); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(right + 7, base); ctx.lineTo(right + 70, base); ctx.stroke();
    ctx.fillStyle = '#f3eee6'; ctx.strokeStyle = '#ddd2c3'; ctx.lineWidth = 1.3;
    ctx.fillRect(43, 515, 270, 82); ctx.strokeRect(43, 515, 270, 82);
    ctx.fillStyle = palette.muted; ctx.font = '750 12px Segoe UI, Arial'; ctx.textAlign = 'left'; ctx.fillText('GIVEN INPUTS', 62, 538);
    ctx.fillStyle = palette.text; ctx.font = '750 16px Segoe UI, Arial';
    ctx.fillText('Sᵤ = $SU kPa     γ = $GAMMA kN/m³', 62, 565); ctx.fillText('B₁ = B/√2 = $B1 m', 62, 589);
    const arrowAlpha = 0.35 + 0.65 * phase; ctx.globalAlpha = arrowAlpha;
    drawArrow(left - 82, ground + 72, left - 82, ground + 132, palette.rose, 3);
    drawArrow(right + 82, ground + 72, right + 82, ground + 132, palette.rose, 3);
    drawArrow(cx, base + 78, cx, base + 14, palette.rose, 3); ctx.globalAlpha = 1;
  }

  function draw(time) {
    const phase = animate ? 0.5 - 0.5 * Math.cos((time / 5100) * Math.PI * 2) : 1.0;
    ctx.clearRect(0, 0, W, Hc); ctx.fillStyle = palette.bg; ctx.fillRect(0, 0, W, Hc);
    const grad = ctx.createLinearGradient(0, ground, 0, Hc); grad.addColorStop(0, palette.soil); grad.addColorStop(1, palette.soilDeep);
    ctx.fillStyle = grad; ctx.fillRect(28, ground, W - 56, Hc - ground - 20);
    drawMesh(phase); drawFailureEnvelope(phase, time); drawStructure(phase); drawSurface(phase); drawLoads(); drawLabels(phase);
    if (animate) requestAnimationFrame(draw);
  }
  if (animate) requestAnimationFrame(draw); else draw(0);
})();
</script>
</body>
</html>
''')

    return template.substitute(
        EXC_W=f"{excavation_width:.2f}",
        ANIMATE="true" if animate else "false",
        Q=f"{q:.1f}",
        H=f"{H:.2f}",
        B=f"{B:.2f}",
        SU=f"{Su:.1f}",
        GAMMA=f"{gamma:.1f}",
        B1=f"{b1:.2f}",
    )



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


st.subheader("Bjerrum & Eide N₍c₎ graph")
components.html(
    build_nc_chart(
        h_over_b=result["H_over_B"],
        nc=result["Nc"],
    ),
    height=590,
    scrolling=False,
)
st.caption(
    "The graph is generated directly from the same piecewise N₍c₎ equations used "
    "in the calculation. The highlighted point updates with the entered H/B ratio."
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

st.subheader("Excavation section and Optum-style basal-heave animation")
components.html(
    build_excavation_animation(
        H=H,
        B=B,
        q=q,
        Su=Su,
        gamma=gamma,
        animate=animate_mechanism,
    ),
    height=720,
    scrolling=False,
)

st.markdown(
    """
    <div class="method-note">
    <b>Animation note:</b> the triangular mesh and symmetric failure envelope are
    styled after the supplied Optum video. The motion illustrates adjacent-ground
    settlement, inward soil movement, and upward base heave; it is not a calculated
    displacement contour or time-history result.
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
