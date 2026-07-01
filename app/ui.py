"""Komponen UI bersama untuk aplikasi Streamlit — tema klinis/profesional.

Dipakai semua halaman agar konsisten: header, badge risiko, chart glukosa dengan
band zona klinis, kartu metrik, dan footer disclaimer. Warna mengikuti makna klinis
(hijau=target, merah=hipoglikemia, oranye=hiperglikemia).
"""
from __future__ import annotations

from typing import Optional, Sequence

import streamlit as st

# ── Ambang & warna klinis (ADA) ───────────────────────────────
GLUCOSE_LOW = 70
GLUCOSE_HIGH = 180

PRIMARY = "#0e7c86"      # teal medis
COL_HYPO = "#d64545"     # merah — hipoglikemia
COL_TARGET = "#2e9e5b"   # hijau — target
COL_HYPER = "#e08a1e"    # oranye — hiperglikemia
COL_INK = "#14303a"      # teks gelap


def classify_glucose(value: float) -> tuple[str, str, str]:
    """Return (kode_zona, label_klinis, warna) dari nilai glukosa."""
    if value < GLUCOSE_LOW:
        return "hipo", "Hipoglikemia", COL_HYPO
    if value > GLUCOSE_HIGH:
        return "hiper", "Hiperglikemia", COL_HYPER
    return "target", "Dalam Target", COL_TARGET


def inject_global_css() -> None:
    """Suntik CSS global (kartu, badge, header, footer). Panggil sekali per halaman."""
    st.markdown(
        f"""
        <style>
        :root {{ --primary: {PRIMARY}; --ink: {COL_INK}; }}
        .block-container {{ padding-top: 1.6rem; }}
        .app-header {{
            display: flex; align-items: center; gap: 0.9rem;
            padding: 1.05rem 1.3rem; border-radius: 14px; margin-bottom: 1.1rem;
            background: linear-gradient(115deg, #0e3b43 0%, #0e7c86 60%, #1a9aa6 100%);
            color: #f6fbfc;
        }}
        .app-header .icon {{ font-size: 2rem; line-height: 1; }}
        .app-header h1 {{ margin: 0; font-size: 1.5rem; letter-spacing: .2px; }}
        .app-header p {{ margin: .2rem 0 0 0; opacity: .92; font-size: .95rem; }}
        .card {{
            border: 1px solid rgba(14,124,134,.18); border-radius: 14px;
            padding: 1rem 1.15rem; background: #ffffff;
            box-shadow: 0 1px 3px rgba(20,48,58,.06);
        }}
        .card h4 {{ margin: 0 0 .35rem 0; color: var(--ink); font-size: 1rem; }}
        .card p {{ margin: 0; color: #33525e; font-size: .92rem; }}
        .risk-badge {{
            display: inline-block; padding: .5rem 1rem; border-radius: 999px;
            font-weight: 700; font-size: 1.02rem; color: #fff; letter-spacing: .3px;
        }}
        .disclaimer {{
            margin-top: 1.4rem; padding: .7rem 1rem; border-left: 4px solid var(--primary);
            background: #eef4f6; border-radius: 8px; color: #33525e; font-size: .85rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def app_header(title: str, subtitle: str = "", icon: str = "🩺") -> None:
    """Header konsisten di tiap halaman."""
    inject_global_css()
    st.markdown(
        f"""
        <div class="app-header">
          <div class="icon">{icon}</div>
          <div>
            <h1>{title}</h1>
            {f'<p>{subtitle}</p>' if subtitle else ''}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_badge(glucose: float, prefix: str = "Prediksi") -> None:
    """Badge risiko besar berwarna sesuai zona klinis."""
    _, label, color = classify_glucose(glucose)
    st.markdown(
        f'<span class="risk-badge" style="background:{color}">'
        f'{prefix}: {label} · {glucose:.0f} mg/dL</span>',
        unsafe_allow_html=True,
    )


def glucose_zone_chart(
    x: Sequence,
    y: Sequence[float],
    predicted_value: Optional[float] = None,
    predicted_x=None,
    title: str = "Tren Glukosa",
    height: int = 340,
):
    """Chart Plotly: garis glukosa + band zona klinis + titik prediksi (opsional)."""
    import plotly.graph_objects as go

    fig = go.Figure()
    # Band zona klinis (latar)
    fig.add_hrect(y0=0, y1=GLUCOSE_LOW, fillcolor=COL_HYPO, opacity=0.08, line_width=0)
    fig.add_hrect(y0=GLUCOSE_LOW, y1=GLUCOSE_HIGH, fillcolor=COL_TARGET, opacity=0.08, line_width=0)
    fig.add_hrect(y0=GLUCOSE_HIGH, y1=400, fillcolor=COL_HYPER, opacity=0.08, line_width=0)
    fig.add_hline(y=GLUCOSE_LOW, line_dash="dot", line_color=COL_HYPO, opacity=0.5)
    fig.add_hline(y=GLUCOSE_HIGH, line_dash="dot", line_color=COL_HYPER, opacity=0.5)

    fig.add_trace(go.Scatter(x=list(x), y=list(y), mode="lines+markers",
                             name="Glukosa", line=dict(color=PRIMARY, width=2.5),
                             marker=dict(size=5)))
    if predicted_value is not None:
        px = predicted_x if predicted_x is not None else (list(x)[-1] if len(x) else 0)
        _, plabel, pcolor = classify_glucose(predicted_value)
        fig.add_trace(go.Scatter(x=[px], y=[predicted_value], mode="markers+text",
                                 name="Prediksi", text=[f"{predicted_value:.0f}"],
                                 textposition="top center",
                                 marker=dict(size=14, color=pcolor, symbol="star",
                                             line=dict(width=1, color="#fff"))))
    fig.update_layout(
        title=title, height=height, margin=dict(l=10, r=10, t=40, b=10),
        yaxis_title="mg/dL", plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(color=COL_INK),
    )
    fig.update_yaxes(range=[40, 400], gridcolor="#eef2f4")
    fig.update_xaxes(gridcolor="#eef2f4")
    return fig


def disclaimer_footer() -> None:
    """Disclaimer klinis konsisten di semua halaman."""
    st.markdown(
        '<div class="disclaimer">⚕️ <b>Catatan:</b> Sistem ini adalah alat bantu '
        'pengambilan keputusan (<i>decision support</i>), bukan pengganti penilaian klinis. '
        'Keputusan medis final tetap pada dokter.</div>',
        unsafe_allow_html=True,
    )


def zone_legend() -> None:
    """Legenda kecil makna warna zona."""
    st.markdown(
        f'<div style="font-size:.82rem;color:#33525e">'
        f'<span style="color:{COL_HYPO}">●</span> Hipoglikemia &lt;70 &nbsp; '
        f'<span style="color:{COL_TARGET}">●</span> Target 70–180 &nbsp; '
        f'<span style="color:{COL_HYPER}">●</span> Hiperglikemia &gt;180</div>',
        unsafe_allow_html=True,
    )
