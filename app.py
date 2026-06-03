#!/usr/bin/env python3
"""PimaGuard - Sistema de Avaliacao de Risco para Diabetes Tipo 2"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import streamlit as st

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PimaGuard",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "PimaGuard — Prototipo de IA para Prevencao de Diabetes Tipo 2"},
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

[data-testid="stAppViewContainer"] { background: #F1F5F9; }
[data-testid="stHeader"] { display: none; }
.block-container { padding: 2rem 2.5rem !important; max-width: 1380px; }
footer { display: none; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0B1D35;
    border-right: 1px solid #152d50;
    min-width: 240px !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span:not(.brand):not(.brand-sub) {
    color: #8DA2BC !important;
    font-size: 0.875rem;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #E2E8F0 !important; }

/* Navigation radio */
.stRadio > label { display: none; }
.stRadio > div { flex-direction: column; gap: 2px; }
.stRadio > div > label {
    padding: 9px 14px;
    border-radius: 7px;
    color: #8DA2BC !important;
    font-size: 0.875rem;
    font-weight: 500;
    transition: all 0.15s;
    cursor: pointer;
    border: 1px solid transparent;
}
.stRadio > div > label:hover {
    background: rgba(255,255,255,0.06);
    color: #CBD5E1 !important;
}
[data-baseweb="radio"] > div:first-child { display: none !important; }

/* ── Cards ── */
.card {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    margin-bottom: 1rem;
}

.kpi-card {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    border-left: 4px solid #0D9488;
    height: 100%;
}
.kpi-card.blue   { border-left-color: #3B82F6; }
.kpi-card.amber  { border-left-color: #F59E0B; }
.kpi-card.red    { border-left-color: #EF4444; }
.kpi-card.green  { border-left-color: #10B981; }
.kpi-card.teal   { border-left-color: #0D9488; }

.kpi-value { font-size: 1.9rem; font-weight: 800; color: #1E293B; line-height: 1.1; }
.kpi-label {
    font-size: 0.72rem; color: #64748B;
    text-transform: uppercase; letter-spacing: 0.08em;
    margin-top: 5px;
}
.kpi-sub { font-size: 0.8rem; color: #94A3B8; margin-top: 2px; }

/* ── Risk badges ── */
.badge {
    display: inline-block;
    padding: 5px 14px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.82rem;
    letter-spacing: 0.02em;
}
.badge-alto     { background: #FEE2E2; color: #DC2626; }
.badge-moderado { background: #FEF3C7; color: #B45309; }
.badge-baixo    { background: #D1FAE5; color: #047857; }

/* ── Section headers ── */
.page-title {
    font-size: 1.6rem; font-weight: 800;
    color: #0F172A; margin-bottom: 2px;
    letter-spacing: -0.02em;
}
.page-subtitle { font-size: 0.9rem; color: #64748B; margin-bottom: 1.5rem; }

.section-title {
    font-size: 1rem; font-weight: 700;
    color: #1E293B; margin-bottom: 0.25rem;
    text-transform: uppercase; letter-spacing: 0.05em;
}

/* ── Divider ── */
.divider { height: 1px; background: #E2E8F0; margin: 1.25rem 0; }

/* ── Sidebar brand ── */
.brand {
    font-size: 1.55rem; font-weight: 900;
    color: #FFFFFF !important;
    letter-spacing: -0.03em; margin-bottom: 0; line-height: 1;
}
.brand-sub {
    font-size: 0.65rem; font-weight: 600;
    color: #0D9488 !important;
    text-transform: uppercase; letter-spacing: 0.15em;
    margin-bottom: 0;
}

/* ── Info / alert boxes ── */
.box-info {
    background: #EFF6FF; border-left: 3px solid #3B82F6;
    padding: 0.7rem 1rem; border-radius: 0 8px 8px 0;
    font-size: 0.85rem; color: #1D4ED8; margin: 0.75rem 0;
}
.box-warning {
    background: #FFFBEB; border-left: 3px solid #F59E0B;
    padding: 0.7rem 1rem; border-radius: 0 8px 8px 0;
    font-size: 0.85rem; color: #92400E; margin: 0.75rem 0;
}
.box-success {
    background: #ECFDF5; border-left: 3px solid #10B981;
    padding: 0.7rem 1rem; border-radius: 0 8px 8px 0;
    font-size: 0.85rem; color: #065F46; margin: 0.75rem 0;
}

/* ── Metrics table ── */
.metrics-table { width: 100%; border-collapse: collapse; font-size: 0.875rem; }
.metrics-table th {
    background: #F8FAFC; color: #475569; font-weight: 600;
    padding: 10px 14px; text-align: left;
    border-bottom: 2px solid #E2E8F0; white-space: nowrap;
}
.metrics-table td {
    padding: 10px 14px; border-bottom: 1px solid #F1F5F9; color: #374151;
}
.metrics-table tr:last-child td { border-bottom: none; }
.metrics-table tr:hover td { background: #F8FAFC; }
.best { font-weight: 700; color: #047857; }

/* ── Buttons ── */
.stButton > button {
    background: #0D9488 !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; padding: 0.6rem 1.5rem !important;
    font-weight: 600 !important; width: 100% !important;
    font-size: 0.9rem !important; transition: background 0.2s !important;
}
.stButton > button:hover { background: #0F766E !important; }

/* ── Sliders ── */
.stSlider > div > div > div > div { background: #0D9488 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background: transparent; gap: 4px; }
.stTabs [data-baseweb="tab"] {
    background: #FFFFFF; border-radius: 8px 8px 0 0;
    border: 1px solid #E2E8F0; border-bottom: none;
    font-weight: 500; color: #64748B; padding: 8px 18px;
}
.stTabs [aria-selected="true"] {
    background: #0D9488 !important; color: white !important;
    border-color: #0D9488 !important;
}

/* ── Sidebar meta ── */
.sidebar-meta {
    background: rgba(255,255,255,0.04);
    border-radius: 8px; padding: 10px 14px;
    margin: 8px 0;
}
.sidebar-meta p { font-size: 0.78rem !important; margin: 2px 0; }

/* ── Plotly chart text — forcar legibilidade ── */
.js-plotly-plot .xtick text,
.js-plotly-plot .ytick text,
.js-plotly-plot .g-xtitle text,
.js-plotly-plot .g-ytitle text,
.js-plotly-plot .legendtext,
.js-plotly-plot .annotation-text tspan {
    fill: #1E293B !important;
    font-size: 12px !important;
}
.js-plotly-plot .gtitle { fill: #0F172A !important; }

/* Tabelas HTML customizadas */
.metrics-table th { color: #1E293B !important; }
.metrics-table td { color: #1E293B !important; }

/* ── Override Streamlit exception/error pink boxes ── */
.stException, [data-testid="stException"] {
    background: #FFFBEB !important;
    border: 1px solid #FDE68A !important;
    border-left: 4px solid #F59E0B !important;
    border-radius: 0 8px 8px 0 !important;
    padding: 0.9rem 1.1rem !important;
}
.stException .message, [data-testid="stException"] .message {
    color: #92400E !important;
    font-weight: 600 !important;
}
.stException pre, [data-testid="stException"] pre {
    color: #78350F !important;
    background: rgba(0,0,0,0.04) !important;
    border-radius: 4px !important;
    padding: 0.5rem !important;
    font-size: 0.72rem !important;
    max-height: 120px;
    overflow-y: auto;
}
/* Override Streamlit error alert */
div[data-baseweb="notification"][kind="negative"],
div[role="alert"].stAlert {
    background: #FFFBEB !important;
    border-left-color: #F59E0B !important;
}

/* ── Slider labels ── */
.stSlider > label {
    font-weight: 600 !important;
    font-size: 0.84rem !important;
    color: #1E293B !important;
}
.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"] {
    color: #94A3B8 !important;
    font-size: 0.7rem !important;
}

/* ── Feature card in prediction form ── */
.feature-card {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 0.7rem 0.9rem 0.4rem;
    margin-bottom: 0.6rem;
}
.feature-name {
    font-size: 0.83rem; font-weight: 700;
    color: #1E293B; margin: 0 0 1px;
}
.feature-desc {
    font-size: 0.72rem; color: #64748B; margin: 0 0 2px;
}
.feature-ref {
    font-size: 0.68rem; color: #0D9488;
    font-weight: 500; margin: 0;
}
.feature-card .stSlider { margin-top: 4px !important; }
</style>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────────────
FEATURE_NAMES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

FEATURE_LABELS = {
    'Pregnancies': 'Gestacoes', 'Glucose': 'Glicose',
    'BloodPressure': 'Pressao Arterial', 'SkinThickness': 'Espessura da Pele',
    'Insulin': 'Insulina', 'BMI': 'IMC',
    'DiabetesPedigreeFunction': 'Pedigree Diabetes', 'Age': 'Idade',
}
FEATURE_UNITS = {
    'Pregnancies': '', 'Glucose': 'mg/dL', 'BloodPressure': 'mmHg',
    'SkinThickness': 'mm', 'Insulin': 'uU/mL', 'BMI': 'kg/m2',
    'DiabetesPedigreeFunction': '', 'Age': 'anos',
}
FEATURE_RANGES = {
    'Pregnancies':             (0,    17,   3,    1),
    'Glucose':                 (44,   199,  117,  1),
    'BloodPressure':           (24,   122,  72,   1),
    'SkinThickness':           (7,    99,   29,   1),
    'Insulin':                 (14,   846,  79,   1),
    'BMI':                     (18.0, 67.1, 32.0, 0.1),
    'DiabetesPedigreeFunction':(0.078,2.42, 0.47, 0.001),
    'Age':                     (21,   81,   33,   1),
}

FEATURE_DESCRIPTIONS = {
    'Pregnancies':              'Numero de gravidezes anteriores',
    'Glucose':                  'Glicose plasmatica 2h apos teste oral de tolerancia',
    'BloodPressure':            'Pressao arterial diastolica (valor minimo do ciclo cardiaco)',
    'SkinThickness':            'Espessura da dobra cutanea do triceps (estimativa de gordura corporal)',
    'Insulin':                  'Nivel de insulina serica 2h apos ingestao de glicose',
    'BMI':                      'Indice de Massa Corporal — peso (kg) dividido por altura² (m)',
    'DiabetesPedigreeFunction': 'Funcao que estima o risco genetico com base no historico familiar',
    'Age':                      'Idade da paciente em anos',
}

FEATURE_REFERENCES = {
    'Pregnancies':              'Referencia: 0 a 5 gestacoes — acima de 6 eleva o risco',
    'Glucose':                  'Normal: < 100 mg/dL | Pre-diabetes: 100-125 | Diabetes: >= 126',
    'BloodPressure':            'Normal: 60-80 mmHg | Elevada: 80-89 | Hipertensao: >= 90',
    'SkinThickness':            'Normal: 10-25 mm | Valores altos indicam maior gordura corporal',
    'Insulin':                  'Jejum normal: 16-166 uU/mL | Valores altos: resistencia a insulina',
    'BMI':                      'Saudavel: 18.5-24.9 | Sobrepeso: 25-29.9 | Obesidade: >= 30',
    'DiabetesPedigreeFunction': 'Quanto maior o valor, maior o risco genetico estimado',
    'Age':                      'Risco aumenta progressivamente a partir dos 45 anos',
}

COLORS = {
    'teal': '#0D9488', 'navy': '#1B3A5C', 'red': '#EF4444',
    'amber': '#F59E0B', 'green': '#10B981', 'blue': '#3B82F6',
    'gray': '#94A3B8', 'light': '#F1F5F9', 'dark': '#1E293B',
}

PLOT_TEMPLATE = dict(
    paper_bgcolor='white', plot_bgcolor='white',
    font=dict(family='Inter', color='#1E293B', size=12),
)

AXIS_STYLE = dict(gridcolor='#F3F4F6', linecolor='#E2E8F0')


# ── System initialization ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Inicializando PimaGuard — treinando modelos, aguarde (~30s na primeira execucao)...")
def init_system():
    from src.data_loader import load_data
    from src.preprocessing import preprocess, split_data, get_clean_df
    from src.train import train_sklearn, load_models, models_need_training, MODELS_DIR
    from src.evaluate import evaluate_all, cross_validate_sklearn

    df = load_data()
    X, y, scaler = preprocess(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    os.makedirs(MODELS_DIR, exist_ok=True)
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    if not os.path.exists(scaler_path):
        joblib.dump(scaler, scaler_path)

    models = load_models()
    if models_need_training():
        new_models = train_sklearn(X_train, y_train)
        models.update(new_models)
    if not models:
        models = train_sklearn(X_train, y_train)

    results = evaluate_all(models, X_test, y_test)

    cv_results = {}
    for name, model in models.items():
        if 'Keras' not in name:
            try:
                cv_results[name] = cross_validate_sklearn(model, X, y)
            except Exception:
                pass

    df_clean = get_clean_df(df)

    return {
        'df': df,
        'df_clean': df_clean,
        'X': X, 'y': y,
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'scaler': scaler,
        'models': models,
        'results': results,
        'cv_results': cv_results,
    }


# ── Chart helpers ─────────────────────────────────────────────────────────────
def risk_gauge(probability: float) -> go.Figure:
    pct = probability * 100
    if pct < 30:
        bar_color, step_idx = COLORS['green'], 0
    elif pct < 60:
        bar_color, step_idx = COLORS['amber'], 1
    else:
        bar_color, step_idx = COLORS['red'], 2

    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=pct,
        number=dict(
            suffix='%',
            font=dict(size=42, color=bar_color, family='Inter'),
        ),
        title=dict(
            text='Probabilidade de Diabetes',
            font=dict(size=13, color='#64748B'),
        ),
        gauge=dict(
            axis=dict(
                range=[0, 100],
                tickvals=[0, 30, 60, 100],
                ticktext=['0%', '30%', '60%', '100%'],
                tickwidth=1, tickcolor='#CBD5E1',
                tickfont=dict(color='#1E293B', size=11),
            ),
            bar=dict(color=bar_color, thickness=0.28),
            bgcolor='white',
            borderwidth=0,
            steps=[
                dict(range=[0, 30], color='#D1FAE5'),
                dict(range=[30, 60], color='#FEF3C7'),
                dict(range=[60, 100], color='#FEE2E2'),
            ],
            threshold=dict(
                line=dict(color=bar_color, width=3),
                thickness=0.8,
                value=pct,
            ),
        ),
    ))
    fig.update_layout(
        height=300,
        paper_bgcolor='white',
        margin=dict(l=30, r=30, t=60, b=10),
        font=dict(family='Inter'),
    )
    return fig


def shap_chart(shap_values: np.ndarray, feature_names: list) -> go.Figure:
    labels = [FEATURE_LABELS.get(f, f) for f in feature_names]
    df = pd.DataFrame({'label': labels, 'shap': shap_values})
    df = df.reindex(df['shap'].abs().sort_values().index)

    colors = [COLORS['red'] if v > 0 else COLORS['blue'] for v in df['shap']]
    texts = [f"{v:+.3f}" for v in df['shap']]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df['label'], x=df['shap'], orientation='h',
        marker_color=colors,
        text=texts, textposition='outside',
        cliponaxis=False,
        hovertemplate='<b>%{y}</b><br>Impacto: %{x:+.4f}<extra></extra>',
    ))
    fig.add_vline(x=0, line_width=1.5, line_color='#374151')

    x_range = max(abs(df['shap'].min()), abs(df['shap'].max())) * 1.5
    tick_font = dict(color='#1E293B', size=12)
    fig.update_layout(
        **PLOT_TEMPLATE,
        height=320,
        xaxis=dict(range=[-x_range, x_range], title='Impacto no Risco (valor SHAP)',
                   title_font=dict(color='#1E293B', size=12),
                   tickfont=tick_font,
                   gridcolor='#F3F4F6', linecolor='#E2E8F0', zeroline=False),
        yaxis=dict(tickfont=tick_font, gridcolor='#F3F4F6', linecolor='#E2E8F0'),
        margin=dict(l=20, r=80, t=35, b=20),
        annotations=[
            dict(x=x_range * 0.7, y=1.04, xref='x', yref='paper',
                 text='<b>Aumenta risco</b>', font=dict(color=COLORS['red'], size=12),
                 showarrow=False, align='center'),
            dict(x=-x_range * 0.7, y=1.04, xref='x', yref='paper',
                 text='<b>Reduz risco</b>', font=dict(color=COLORS['blue'], size=12),
                 showarrow=False, align='center'),
        ],
    )
    return fig


def pr_chart(results: dict) -> go.Figure:
    fig = go.Figure()
    palette = [COLORS['teal'], COLORS['blue'], COLORS['amber'], COLORS['red'], '#8B5CF6']
    # Baseline (prevalencia)
    pos_rate = 268 / 768
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[pos_rate, pos_rate], mode='lines',
        line=dict(dash='dash', color='#CBD5E1', width=1.5),
        name=f'Baseline ({pos_rate:.2f})', showlegend=True,
    ))
    for i, (name, data) in enumerate(results.items()):
        prec, rec = data['pr_curve']
        auc = data['metrics']['PR-AUC']
        fig.add_trace(go.Scatter(
            x=rec, y=prec, mode='lines',
            name=f"{name} (AP={auc:.3f})",
            line=dict(width=2.5, color=palette[i % len(palette)]),
            hovertemplate='Recall: %{x:.3f}<br>Precisao: %{y:.3f}<extra></extra>',
        ))
    fig.update_layout(
        **PLOT_TEMPLATE, height=420,
        xaxis=dict(title='Recall (Sensibilidade)', range=[0, 1],
                   gridcolor='#F3F4F6', linecolor='#E2E8F0'),
        yaxis=dict(title='Precisao', range=[0, 1.02],
                   gridcolor='#F3F4F6', linecolor='#E2E8F0', tickfont=dict(color='#1E293B', size=11)),
        legend=dict(x=0.01, y=0.05, bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='#E2E8F0', borderwidth=1, font=dict(size=11)),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


def calibration_chart(results: dict) -> go.Figure:
    fig = go.Figure()
    palette = [COLORS['teal'], COLORS['blue'], COLORS['amber'], COLORS['red'], '#8B5CF6']
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines',
        line=dict(dash='dash', color='#CBD5E1', width=1.5),
        name='Calibracao perfeita', showlegend=True,
    ))
    for i, (name, data) in enumerate(results.items()):
        mean_pred, frac_pos = data['calibration']
        if len(mean_pred) == 0:
            continue
        fig.add_trace(go.Scatter(
            x=mean_pred, y=frac_pos, mode='lines+markers',
            name=name,
            line=dict(width=2.5, color=palette[i % len(palette)]),
            marker=dict(size=8),
            hovertemplate='Prob. prevista: %{x:.3f}<br>Frac. positivos: %{y:.3f}<extra></extra>',
        ))
    fig.update_layout(
        **PLOT_TEMPLATE, height=420,
        xaxis=dict(title='Probabilidade media prevista', range=[0, 1],
                   gridcolor='#F3F4F6', linecolor='#E2E8F0', tickfont=dict(color='#1E293B', size=11)),
        yaxis=dict(title='Fracao de positivos reais', range=[0, 1.02],
                   gridcolor='#F3F4F6', linecolor='#E2E8F0', tickfont=dict(color='#1E293B', size=11)),
        legend=dict(x=0.01, y=0.95, bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='#E2E8F0', borderwidth=1, font=dict(size=11)),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


def roc_chart(results: dict) -> go.Figure:
    fig = go.Figure()
    palette = [COLORS['teal'], COLORS['blue'], COLORS['amber'], COLORS['red'], '#8B5CF6']
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines',
        line=dict(dash='dash', color='#CBD5E1', width=1.5),
        name='Referencia aleatoria', showlegend=True,
    ))
    for i, (name, data) in enumerate(results.items()):
        fpr, tpr = data['roc']
        auc = data['metrics']['ROC-AUC']
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode='lines',
            name=f"{name} (AUC={auc:.3f})",
            line=dict(width=2.5, color=palette[i % len(palette)]),
            hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>',
        ))
    fig.update_layout(
        **PLOT_TEMPLATE,
        height=420,
        xaxis=dict(title='Taxa de Falso Positivo', range=[0, 1],
                   gridcolor='#F3F4F6', linecolor='#E2E8F0', tickfont=dict(color='#1E293B', size=11)),
        yaxis=dict(title='Taxa de Verdadeiro Positivo', range=[0, 1.02],
                   gridcolor='#F3F4F6', linecolor='#E2E8F0', tickfont=dict(color='#1E293B', size=11)),
        legend=dict(
            x=0.52, y=0.05, bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#E2E8F0', borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


def confusion_matrix_chart(cm: np.ndarray, model_name: str) -> go.Figure:
    labels = ['Nao-diabetico', 'Diabetico']
    total = cm.sum()
    text = [[f"{cm[i][j]}<br>({cm[i][j]/total*100:.1f}%)" for j in range(2)] for i in range(2)]

    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale=[[0, '#EFF6FF'], [1, '#1D4ED8']],
        showscale=False, text=text, texttemplate='%{text}',
        hovertemplate='Real: %{y}<br>Previsto: %{x}<br>Count: %{z}<extra></extra>',
        textfont=dict(size=15, family='Inter'),
    ))
    fig.update_layout(
        **PLOT_TEMPLATE,
        height=300,
        xaxis=dict(title='Previsto', side='bottom', gridcolor='rgba(0,0,0,0)', tickfont=dict(color='#1E293B', size=11), linecolor='rgba(0,0,0,0)'),
        yaxis=dict(title='Real', gridcolor='rgba(0,0,0,0)', tickfont=dict(color='#1E293B', size=11), linecolor='rgba(0,0,0,0)', autorange='reversed'),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


def population_comparison_chart(patient_values: list, df_clean: pd.DataFrame) -> go.Figure:
    df0 = df_clean[df_clean['Outcome'] == 0]
    df1 = df_clean[df_clean['Outcome'] == 1]

    categories = [FEATURE_LABELS[f] for f in FEATURE_NAMES]

    def normalize(vals, mins, maxs):
        return [(v - mn) / max(mx - mn, 1e-9) for v, mn, mx in zip(vals, mins, maxs)]

    mins = [df_clean[f].min() for f in FEATURE_NAMES]
    maxs = [df_clean[f].max() for f in FEATURE_NAMES]

    patient_norm = normalize(patient_values, mins, maxs)
    pop0_norm = normalize([df0[f].mean() for f in FEATURE_NAMES], mins, maxs)
    pop1_norm = normalize([df1[f].mean() for f in FEATURE_NAMES], mins, maxs)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=pop0_norm + [pop0_norm[0]], theta=categories + [categories[0]],
        mode='lines', name='Media nao-diabetico',
        line=dict(color=COLORS['blue'], width=2, dash='dot'),
        fill='toself', fillcolor='rgba(59,130,246,0.07)',
    ))
    fig.add_trace(go.Scatterpolar(
        r=pop1_norm + [pop1_norm[0]], theta=categories + [categories[0]],
        mode='lines', name='Media diabetico',
        line=dict(color=COLORS['red'], width=2, dash='dot'),
        fill='toself', fillcolor='rgba(239,68,68,0.07)',
    ))
    fig.add_trace(go.Scatterpolar(
        r=patient_norm + [patient_norm[0]], theta=categories + [categories[0]],
        mode='lines+markers', name='Paciente',
        line=dict(color=COLORS['teal'], width=3),
        marker=dict(size=7, color=COLORS['teal']),
        fill='toself', fillcolor='rgba(13,148,136,0.12)',
    ))
    fig.update_layout(
        paper_bgcolor='white',
        polar=dict(
            bgcolor='white',
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False,
                            gridcolor='#E2E8F0', linecolor='#E2E8F0'),
            angularaxis=dict(gridcolor='#E2E8F0', linecolor='#E2E8F0',
                             tickfont=dict(color='#1E293B', size=11)),
        ),
        legend=dict(x=0.5, y=-0.15, xanchor='center', orientation='h',
                    font=dict(size=11)),
        margin=dict(l=30, r=30, t=20, b=60),
        height=380,
        font=dict(family='Inter', color='#374151'),
    )
    return fig


def metrics_comparison_chart(results: dict) -> go.Figure:
    metric_keys = ['Acuracia', 'Precisao', 'Recall', 'F1-Score', 'ROC-AUC']
    palette = [COLORS['teal'], COLORS['blue'], COLORS['amber'], COLORS['red'], '#8B5CF6']
    models_list = list(results.keys())

    fig = go.Figure()
    x_pos = np.arange(len(metric_keys))
    bar_width = 0.8 / len(models_list)

    for i, (name, data) in enumerate(results.items()):
        vals = [data['metrics'][k] for k in metric_keys]
        offset = (i - len(models_list) / 2 + 0.5) * bar_width
        fig.add_trace(go.Bar(
            x=[xi + offset for xi in x_pos], y=vals,
            name=name, marker_color=palette[i % len(palette)],
            width=bar_width * 0.9,
            hovertemplate='<b>%{fullData.name}</b><br>%{x}<br>%{y:.4f}<extra></extra>',
        ))

    fig.update_layout(
        **PLOT_TEMPLATE,
        height=380,
        xaxis=dict(
            tickvals=list(x_pos), ticktext=metric_keys,
            gridcolor='#F3F4F6', linecolor='#E2E8F0',
        ),
        yaxis=dict(range=[0, 1.05], title='Valor da Metrica',
                   gridcolor='#F3F4F6', linecolor='#E2E8F0', tickfont=dict(color='#1E293B', size=11)),
        legend=dict(x=0.5, y=1.08, xanchor='center', orientation='h', font=dict(size=11)),
        barmode='group',
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


# ── Pages ─────────────────────────────────────────────────────────────────────
def page_dashboard(sd: dict):
    df = sd['df']
    results = sd['results']

    best_name = max(results, key=lambda n: results[n]['metrics']['ROC-AUC'])
    best = results[best_name]['metrics']
    neg = int((df['Outcome'] == 0).sum())
    pos = int((df['Outcome'] == 1).sum())

    st.markdown('<p class="page-title">Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Visao geral do sistema e desempenho dos modelos treinados</p>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="kpi-card blue">
            <div class="kpi-value">{len(df)}</div>
            <div class="kpi-label">Pacientes no Dataset</div>
            <div class="kpi-sub">Pima Indians Diabetes</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card red">
            <div class="kpi-value">{pos/len(df)*100:.1f}%</div>
            <div class="kpi-label">Taxa de Diabetes</div>
            <div class="kpi-sub">{pos} de {len(df)} pacientes</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card green">
            <div class="kpi-value">{best['ROC-AUC']:.3f}</div>
            <div class="kpi-label">Melhor AUC</div>
            <div class="kpi-sub">{best_name}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-card teal">
            <div class="kpi-value">{best['Acuracia']*100:.1f}%</div>
            <div class="kpi-label">Melhor Acuracia</div>
            <div class="kpi-sub">{best_name}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">Comparacao de Modelos</p>', unsafe_allow_html=True)
        try:
            st.plotly_chart(metrics_comparison_chart(results), use_container_width=True, config={'displayModeBar': False})
        except Exception as exc:
            st.markdown(f'<div class="box-warning">Erro ao gerar grafico: {exc}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">Distribuicao da Classe</p>', unsafe_allow_html=True)
        try:
            fig_donut = go.Figure(go.Pie(
                labels=['Nao-diabetico', 'Diabetico'],
                values=[neg, pos],
                hole=0.55,
                marker_colors=[COLORS['blue'], COLORS['red']],
                textinfo='percent+label',
                hovertemplate='%{label}: %{value} pacientes (%{percent})<extra></extra>',
            ))
            fig_donut.update_layout(
                paper_bgcolor='white', showlegend=False,
                margin=dict(l=10, r=10, t=10, b=10),
                height=220,
                font=dict(family='Inter', size=12),
                annotations=[dict(text=f'{len(df)}<br>total', x=0.5, y=0.5,
                                  font=dict(size=14, color='#374151'), showarrow=False)],
            )
            st.plotly_chart(fig_donut, use_container_width=True, config={'displayModeBar': False})
        except Exception as exc:
            st.markdown(f'<div class="box-warning">Erro ao gerar grafico: {exc}</div>', unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<p class="section-title">Variaveis do Dataset</p>', unsafe_allow_html=True)
        for f in FEATURE_NAMES:
            lab = FEATURE_LABELS[f]
            unit = f" ({FEATURE_UNITS[f]})" if FEATURE_UNITS[f] else ""
            st.markdown(f'<p style="font-size:0.8rem;color:#374151;margin:3px 0">'
                        f'<span style="color:#0D9488;font-weight:600">{lab}</span>{unit}</p>',
                        unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Tabela de Desempenho Completa</p>', unsafe_allow_html=True)
    metric_keys = ['Acuracia', 'Precisao', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC']
    cv = sd.get('cv_results', {})
    header = (
        "<thead><tr><th>Modelo</th>"
        + "".join(f"<th>{k}</th>" for k in metric_keys)
        + "<th>AUC-CV (media ± dp)</th></tr></thead>"
    )
    rows = ""
    for name, data in results.items():
        m = data['metrics']
        cells = []
        for k in metric_keys:
            v = m[k]
            best_v = max(results[n]['metrics'][k] for n in results)
            cls = ' class="best"' if abs(v - best_v) < 1e-6 else ''
            cells.append(f'<td{cls}>{v:.4f}</td>')
        cv_cell = (
            f"{cv[name]['auc_mean']:.4f} ± {cv[name]['auc_std']:.4f}"
            if name in cv else "—"
        )
        star = " *" if name == best_name else ""
        rows += f"<tr><td><b>{name}{star}</b></td>{''.join(cells)}<td>{cv_cell}</td></tr>"
    st.markdown(
        f'<table class="metrics-table"><{header}<tbody>{rows}</tbody></table>'
        '<p style="font-size:0.75rem;color:#94A3B8;margin-top:8px">'
        '* Melhor modelo por ROC-AUC | Negrito: melhor da coluna | '
        'AUC-CV: validacao cruzada 5-Fold (media ± desvio padrao)</p>',
        unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def page_prediction(sd: dict):
    from src.evaluate import predict_proba, compute_shap_values

    models  = sd['models']
    scaler  = sd['scaler']
    X_train = sd['X_train']
    df_clean = sd['df_clean']
    results  = sd['results']
    best_name = max(results, key=lambda n: results[n]['metrics']['ROC-AUC'])

    st.markdown('<p class="page-title">Avaliacao de Risco Clinico</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Insira os dados do paciente para calcular a probabilidade de diabetes tipo 2</p>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="box-warning">Prototipo academico — nao substitui avaliacao medica profissional.</div>',
        unsafe_allow_html=True)

    # ── Controles globais ─────────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 1])
    with ctrl1:
        model_name = st.selectbox(
            "Modelo de predicao",
            options=list(models.keys()),
            index=list(models.keys()).index(best_name) if best_name in models else 0,
            help="Selecione o algoritmo para calcular o risco",
        )
    with ctrl2:
        threshold = st.slider(
            "Limiar de classificacao",
            min_value=0.10, max_value=0.90, value=0.50, step=0.05,
            help="Abaixar o limiar aumenta a sensibilidade (detecta mais casos) "
                 "mas tambem aumenta falsos positivos. Padrao clinico: 0.50",
        )
    with ctrl3:
        st.markdown(
            f'<div style="margin-top:1.8rem;padding:8px 12px;background:#F8FAFC;'
            f'border:1px solid #E2E8F0;border-radius:8px;text-align:center">'
            f'<p style="font-size:0.7rem;color:#64748B;margin:0">Limiar ativo</p>'
            f'<p style="font-size:1.3rem;font-weight:800;color:#0D9488;margin:0">{threshold:.0%}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)

    # ── Abas: Individual | Lote ───────────────────────────────────────────────
    tab_ind, tab_lote = st.tabs(["Paciente Individual", "Predicao em Lote (CSV)"])

    # ── Aba Individual ────────────────────────────────────────────────────────
    with tab_ind:
        form_col, result_col = st.columns([1, 1], gap="large")

        patient_values = {}

        def render_feature_slider(feat):
            mn, mx, default, step = FEATURE_RANGES[feat]
            lab      = FEATURE_LABELS[feat]
            unit     = FEATURE_UNITS[feat]
            unit_txt = f" ({unit})" if unit else ""
            st.markdown(
                f'<div class="feature-card">'
                f'<p class="feature-name">{lab}{unit_txt}</p>'
                f'<p class="feature-desc">{FEATURE_DESCRIPTIONS[feat]}</p>'
                f'<p class="feature-ref">{FEATURE_REFERENCES[feat]}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )
            return st.slider(
                label=f"{lab}{unit_txt}",
                min_value=float(mn), max_value=float(mx),
                value=float(default), step=float(step),
                key=f"slider_{feat}",
                label_visibility="collapsed",
            )

        with form_col:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<p class="section-title">Dados do Paciente</p>', unsafe_allow_html=True)
            fc1, fc2 = st.columns(2)
            left_features  = ['Pregnancies', 'BloodPressure', 'SkinThickness', 'DiabetesPedigreeFunction']
            right_features = ['Glucose', 'Insulin', 'BMI', 'Age']
            with fc1:
                for feat in left_features:
                    patient_values[feat] = render_feature_slider(feat)
            with fc2:
                for feat in right_features:
                    patient_values[feat] = render_feature_slider(feat)
            st.markdown('</div>', unsafe_allow_html=True)

        with result_col:
            patient_array  = np.array([[patient_values[f] for f in FEATURE_NAMES]])
            patient_scaled = scaler.transform(patient_array)
            model = models[model_name]
            prob  = float(predict_proba(model, patient_scaled)[0])
            classified_positive = prob >= threshold

            if prob < 0.30:
                risk_level, badge_cls = "Baixo",    "badge-baixo"
                advice = "Perfil de risco dentro da faixa normal. Mantenha habitos saudaveis e realize exames periodicos."
            elif prob < 0.60:
                risk_level, badge_cls = "Moderado", "badge-moderado"
                advice = "Risco moderado identificado. Recomenda-se avaliacao medica e revisao de habitos alimentares."
            else:
                risk_level, badge_cls = "Alto",     "badge-alto"
                advice = "Risco elevado detectado. Encaminhamento medico urgente recomendado para avaliacao completa."

            decision_color = '#EF4444' if classified_positive else '#10B981'
            decision_label = 'POSITIVO' if classified_positive else 'NEGATIVO'

            st.markdown('<div class="card" style="text-align:center">', unsafe_allow_html=True)
            try:
                st.plotly_chart(risk_gauge(prob), use_container_width=True, config={'displayModeBar': False})
            except Exception:
                pass
            st.markdown(
                f'<p style="text-align:center;margin:4px 0 4px">'
                f'<span class="badge {badge_cls}">Risco {risk_level}</span></p>'
                f'<p style="font-size:0.78rem;color:#94A3B8;margin:4px 0 2px">'
                f'Com limiar de {threshold:.0%}:</p>'
                f'<p style="font-size:1.1rem;font-weight:800;color:{decision_color};margin:0 0 8px">'
                f'{decision_label}</p>'
                f'<p style="font-size:0.83rem;color:#475569;text-align:center;margin-top:4px">{advice}</p>',
                unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── SHAP + Radar ──────────────────────────────────────────────────────
        st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
        shap_col, radar_col = st.columns([1, 1], gap="large")

        with shap_col:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<p class="section-title">Fatores Determinantes (SHAP)</p>', unsafe_allow_html=True)
            st.markdown('<p style="font-size:0.8rem;color:#64748B;margin-bottom:8px">'
                        'Contribuicao de cada variavel para a predicao deste paciente</p>',
                        unsafe_allow_html=True)
            shap_vals = compute_shap_values(model, model_name, X_train, patient_scaled)
            if shap_vals is not None:
                try:
                    st.plotly_chart(shap_chart(shap_vals, FEATURE_NAMES),
                                    use_container_width=True, config={'displayModeBar': False})
                except Exception as exc:
                    st.markdown(f'<div class="box-warning">Erro ao gerar SHAP: {exc}</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="box-warning"><strong>SHAP indisponivel para este modelo.</strong><br>'
                    'Selecione <em>Random Forest</em> ou <em>Gradient Boosting</em>.</div>',
                    unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with radar_col:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<p class="section-title">Comparacao com a Populacao</p>', unsafe_allow_html=True)
            st.markdown('<p style="font-size:0.8rem;color:#64748B;margin-bottom:8px">'
                        'Perfil do paciente versus medias populacionais (valores normalizados)</p>',
                        unsafe_allow_html=True)
            try:
                st.plotly_chart(
                    population_comparison_chart([patient_values[f] for f in FEATURE_NAMES], df_clean),
                    use_container_width=True, config={'displayModeBar': False})
            except Exception as exc:
                st.markdown(f'<div class="box-warning">Erro: {exc}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ── Aba Lote ──────────────────────────────────────────────────────────────
    with tab_lote:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">Predicao em Lote</p>', unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:0.82rem;color:#64748B;margin-bottom:12px">'
            'Faca upload de um arquivo CSV com as colunas: '
            '<code>Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, '
            'DiabetesPedigreeFunction, Age</code></p>',
            unsafe_allow_html=True)

        uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

        if uploaded is not None:
            try:
                df_upload = pd.read_csv(uploaded)
                missing = [c for c in FEATURE_NAMES if c not in df_upload.columns]
                if missing:
                    st.markdown(
                        f'<div class="box-warning">Colunas ausentes no CSV: '
                        f'<strong>{", ".join(missing)}</strong></div>',
                        unsafe_allow_html=True)
                else:
                    X_batch = df_upload[FEATURE_NAMES].fillna(df_upload[FEATURE_NAMES].median())
                    X_batch_scaled = scaler.transform(X_batch.values)
                    batch_model = models[model_name]
                    probas = predict_proba(batch_model, X_batch_scaled)
                    preds  = (probas >= threshold).astype(int)

                    df_result = df_upload[FEATURE_NAMES].copy()
                    df_result['Prob_Diabetes (%)'] = (probas * 100).round(1)
                    df_result['Classificacao']     = np.where(preds == 1, 'POSITIVO', 'NEGATIVO')
                    df_result['Risco']             = pd.cut(
                        probas,
                        bins=[0, 0.3, 0.6, 1.0],
                        labels=['Baixo', 'Moderado', 'Alto'],
                    )

                    n_pos = int(preds.sum())
                    n_neg = len(preds) - n_pos
                    b1, b2, b3 = st.columns(3)
                    with b1:
                        st.markdown(f'<div class="kpi-card blue" style="padding:0.8rem 1rem">'
                                    f'<div class="kpi-value" style="font-size:1.4rem">{len(preds)}</div>'
                                    f'<div class="kpi-label">Total de pacientes</div></div>',
                                    unsafe_allow_html=True)
                    with b2:
                        st.markdown(f'<div class="kpi-card red" style="padding:0.8rem 1rem">'
                                    f'<div class="kpi-value" style="font-size:1.4rem">{n_pos}</div>'
                                    f'<div class="kpi-label">Classificados positivos</div>'
                                    f'<div class="kpi-sub">{n_pos/len(preds)*100:.1f}%</div></div>',
                                    unsafe_allow_html=True)
                    with b3:
                        st.markdown(f'<div class="kpi-card green" style="padding:0.8rem 1rem">'
                                    f'<div class="kpi-value" style="font-size:1.4rem">{n_neg}</div>'
                                    f'<div class="kpi-label">Classificados negativos</div>'
                                    f'<div class="kpi-sub">{n_neg/len(preds)*100:.1f}%</div></div>',
                                    unsafe_allow_html=True)

                    st.markdown('<div style="height:0.75rem"></div>', unsafe_allow_html=True)
                    st.dataframe(
                        df_result.style.apply(
                            lambda col: ['background:#FEE2E2;color:#991B1B' if v == 'POSITIVO'
                                         else 'background:#D1FAE5;color:#065F46'
                                         for v in col]
                            if col.name == 'Classificacao' else ['' for _ in col],
                            axis=0,
                        ),
                        use_container_width=True,
                        height=min(400, 40 + len(df_result) * 36),
                    )

                    csv_bytes = df_result.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Baixar resultados (.csv)",
                        data=csv_bytes,
                        file_name="pimagard_resultados.csv",
                        mime="text/csv",
                    )

            except Exception as exc:
                st.markdown(f'<div class="box-warning">Erro ao processar CSV: <strong>{exc}</strong></div>',
                            unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="box-info">Arraste um arquivo CSV aqui ou clique para selecionar. '
                'O arquivo deve ter as 8 variaveis do modelo nas colunas.</div>',
                unsafe_allow_html=True)
            # Botão para baixar template
            template_df = pd.DataFrame([{f: FEATURE_RANGES[f][2] for f in FEATURE_NAMES}])
            st.download_button(
                label="Baixar template CSV",
                data=template_df.to_csv(index=False).encode('utf-8'),
                file_name="template_pimagard.csv",
                mime="text/csv",
            )
        st.markdown('</div>', unsafe_allow_html=True)


def page_eda(sd: dict):
    df = sd['df']
    df_clean = sd['df_clean']

    st.markdown('<p class="page-title">Analise Exploratoria dos Dados</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Pima Indians Diabetes Database — distribuicoes, correlacoes e padroes</p>',
                unsafe_allow_html=True)

    # Overview cards
    c1, c2, c3, c4 = st.columns(4)
    zero_counts = {f: int((df[f] == 0).sum()) for f in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']}
    total_invalid = sum(zero_counts.values())
    with c1:
        st.markdown(f'<div class="kpi-card blue"><div class="kpi-value">768</div>'
                    f'<div class="kpi-label">Total de Pacientes</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="kpi-card teal"><div class="kpi-value">8</div>'
                    f'<div class="kpi-label">Variaveis Preditoras</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="kpi-card amber"><div class="kpi-value">{total_invalid}</div>'
                    f'<div class="kpi-label">Zeros Invalidos</div>'
                    f'<div class="kpi-sub">substituidos pela mediana por classe</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="kpi-card red"><div class="kpi-value">34.9%</div>'
                    f'<div class="kpi-label">Prevalencia de Diabetes</div></div>', unsafe_allow_html=True)

    st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Distribuicoes", "Correlacao", "Comparacao por Classe", "Dispersao"
    ])

    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        col_sel, col_desc, _ = st.columns([1, 2, 1])
        with col_sel:
            feat_sel = st.selectbox(
                "Variavel", FEATURE_NAMES,
                format_func=lambda x: FEATURE_LABELS[x],
            )
        with col_desc:
            st.markdown(
                f'<p style="font-size:0.78rem;color:#64748B;margin-top:2.1rem;line-height:1.3">'
                f'{FEATURE_DESCRIPTIONS[feat_sel]}</p>',
                unsafe_allow_html=True,
            )

        unit = f" ({FEATURE_UNITS[feat_sel]})" if FEATURE_UNITS[feat_sel] else ""
        try:
            fig = go.Figure()
            for cls, label, color in [(0, 'Nao-diabetico', COLORS['blue']), (1, 'Diabetico', COLORS['red'])]:
                vals = df_clean[df_clean['Outcome'] == cls][feat_sel]
                fig.add_trace(go.Histogram(
                    x=vals, name=label, opacity=0.72,
                    marker_color=color, nbinsx=30,
                    hovertemplate='%{x}<br>Contagem: %{y}<extra></extra>',
                ))
            fig.update_layout(
                paper_bgcolor='white', plot_bgcolor='white',
                font=dict(family='Inter', color='#374151', size=12),
                height=360, barmode='overlay',
                xaxis=dict(title=f"{FEATURE_LABELS[feat_sel]}{unit}",
                           gridcolor='#F3F4F6', linecolor='#E2E8F0', tickfont=dict(color='#1E293B', size=11)),
                yaxis=dict(title='Frequencia', gridcolor='#F3F4F6', linecolor='#E2E8F0', tickfont=dict(color='#1E293B', size=11)),
                legend=dict(x=0.75, y=0.95, font=dict(size=11)),
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        except Exception as exc:
            st.markdown(
                f'<div class="box-warning">Erro ao gerar grafico: {exc}. '
                f'Recarregue a pagina ou selecione outra variavel.</div>',
                unsafe_allow_html=True,
            )

        d0 = df_clean[df_clean['Outcome'] == 0][feat_sel]
        d1 = df_clean[df_clean['Outcome'] == 1][feat_sel]
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            st.markdown(f'<div class="kpi-card blue" style="padding:0.8rem 1rem">'
                        f'<div class="kpi-value" style="font-size:1.3rem">{d0.mean():.2f}</div>'
                        f'<div class="kpi-label">Media — Sem diabetes</div></div>', unsafe_allow_html=True)
        with cc2:
            st.markdown(f'<div class="kpi-card red" style="padding:0.8rem 1rem">'
                        f'<div class="kpi-value" style="font-size:1.3rem">{d1.mean():.2f}</div>'
                        f'<div class="kpi-label">Media — Com diabetes</div></div>', unsafe_allow_html=True)
        with cc3:
            base = d0.mean()
            diff_pct = (d1.mean() - base) / base * 100 if base != 0 else 0
            st.markdown(f'<div class="kpi-card amber" style="padding:0.8rem 1rem">'
                        f'<div class="kpi-value" style="font-size:1.3rem">{diff_pct:+.1f}%</div>'
                        f'<div class="kpi-label">Diferenca entre grupos</div></div>', unsafe_allow_html=True)
        st.markdown(
            f'<p style="font-size:0.72rem;color:#94A3B8;margin:6px 0 0">'
            f'{FEATURE_REFERENCES[feat_sel]}</p>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        try:
            corr = df_clean[FEATURE_NAMES + ['Outcome']].corr()
            labels_pt = [FEATURE_LABELS.get(f, f) for f in FEATURE_NAMES] + ['Diabetes']
            fig_corr = go.Figure(go.Heatmap(
                z=corr.values, x=labels_pt, y=labels_pt,
                colorscale='RdBu', zmid=0, zmin=-1, zmax=1,
                text=[[f"{v:.2f}" for v in row] for row in corr.values],
                texttemplate='%{text}', textfont=dict(size=10),
                hovertemplate='%{y} x %{x}: %{z:.3f}<extra></extra>',
            ))
            fig_corr.update_layout(
                paper_bgcolor='white', plot_bgcolor='white',
                font=dict(family='Inter', color='#374151', size=12),
                height=480,
                xaxis=dict(tickangle=-35, gridcolor='rgba(0,0,0,0)', tickfont=dict(color='#1E293B', size=11), linecolor='rgba(0,0,0,0)'),
                yaxis=dict(gridcolor='rgba(0,0,0,0)', tickfont=dict(color='#1E293B', size=11), linecolor='rgba(0,0,0,0)'),
                margin=dict(l=20, r=20, t=20, b=80),
            )
            st.plotly_chart(fig_corr, use_container_width=True, config={'displayModeBar': False})
        except Exception as exc:
            st.markdown(f'<div class="box-warning">Erro ao gerar mapa de correlacao: {exc}</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="box-info">Glicose apresenta a maior correlacao com Diabetes (r = 0.47), '
            'seguida de IMC (0.29) e Idade (0.24). Insulina tem alta variancia apos imputacao.</div>',
            unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:0.8rem;color:#64748B;margin-bottom:12px">'
            'Comparacao das distribuicoes de cada variavel entre pacientes com e sem diabetes '
            '(valores normalizados para facilitar a comparacao entre escalas diferentes).</p>',
            unsafe_allow_html=True,
        )
        try:
            fig_box = go.Figure()
            for cls, label, color in [(0, 'Nao-diabetico', COLORS['blue']), (1, 'Diabetico', COLORS['red'])]:
                group = df_clean[df_clean['Outcome'] == cls]
                for fi, feat in enumerate(FEATURE_NAMES):
                    vals = group[feat]
                    mn, mx = df_clean[feat].min(), df_clean[feat].max()
                    norm_vals = (vals - mn) / max(mx - mn, 1e-9)
                    fig_box.add_trace(go.Box(
                        y=norm_vals, x=[FEATURE_LABELS[feat]] * len(vals),
                        name=label,
                        legendgroup=label,
                        showlegend=(fi == 0),
                        marker_color=color,
                        boxmean=True,
                        hovertemplate='%{x}<br>%{y:.3f}<extra></extra>',
                    ))
            fig_box.update_layout(
                paper_bgcolor='white', plot_bgcolor='white',
                font=dict(family='Inter', color='#374151', size=12),
                height=420, boxmode='group',
                xaxis=dict(title='Variavel', gridcolor='#F3F4F6', linecolor='#E2E8F0', tickfont=dict(color='#1E293B', size=11)),
                yaxis=dict(title='Valor normalizado [0-1]', gridcolor='#F3F4F6', linecolor='#E2E8F0', tickfont=dict(color='#1E293B', size=11)),
                legend=dict(x=0.85, y=0.98, font=dict(size=11)),
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig_box, use_container_width=True, config={'displayModeBar': False})
        except Exception as exc:
            st.markdown(f'<div class="box-warning">Erro ao gerar grafico: {exc}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        col_x, col_y, _ = st.columns([1, 1, 2])
        with col_x:
            fx = st.selectbox("Eixo X", FEATURE_NAMES, index=1,
                              format_func=lambda x: FEATURE_LABELS[x], key='scatter_x')
        with col_y:
            fy = st.selectbox("Eixo Y", FEATURE_NAMES, index=5,
                              format_func=lambda x: FEATURE_LABELS[x], key='scatter_y')
        try:
            fig_sc = go.Figure()
            for cls, label, color in [(0, 'Nao-diabetico', COLORS['blue']), (1, 'Diabetico', COLORS['red'])]:
                sub = df_clean[df_clean['Outcome'] == cls]
                fig_sc.add_trace(go.Scatter(
                    x=sub[fx], y=sub[fy], mode='markers', name=label,
                    marker=dict(color=color, size=5, opacity=0.65),
                    hovertemplate=f'{FEATURE_LABELS[fx]}: %{{x}}<br>{FEATURE_LABELS[fy]}: %{{y}}<extra></extra>',
                ))
            ux = f" ({FEATURE_UNITS[fx]})" if FEATURE_UNITS[fx] else ""
            uy = f" ({FEATURE_UNITS[fy]})" if FEATURE_UNITS[fy] else ""
            fig_sc.update_layout(
                paper_bgcolor='white', plot_bgcolor='white',
                font=dict(family='Inter', color='#374151', size=12),
                height=420,
                xaxis=dict(title=f"{FEATURE_LABELS[fx]}{ux}", gridcolor='#F3F4F6', linecolor='#E2E8F0', tickfont=dict(color='#1E293B', size=11)),
                yaxis=dict(title=f"{FEATURE_LABELS[fy]}{uy}", gridcolor='#F3F4F6', linecolor='#E2E8F0', tickfont=dict(color='#1E293B', size=11)),
                legend=dict(x=0.75, y=0.98, font=dict(size=11)),
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig_sc, use_container_width=True, config={'displayModeBar': False})
        except Exception as exc:
            st.markdown(f'<div class="box-warning">Erro ao gerar grafico: {exc}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


def page_models(sd: dict):
    results = sd['results']
    cv_results = sd['cv_results']
    X = sd['X']
    y = sd['y']

    st.markdown('<p class="page-title">Comparacao de Modelos</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Desempenho, curvas ROC, matrizes de confusao e validacao cruzada</p>',
                unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Curvas ROC", "Precisao-Recall", "Calibracao",
        "Matrizes de Confusao", "Validacao Cruzada", "Analise de Desempenho"
    ])

    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        try:
            st.plotly_chart(roc_chart(results), use_container_width=True, config={'displayModeBar': False})
        except Exception as exc:
            st.markdown(f'<div class="box-warning">Erro ao gerar curva ROC: {exc}</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="box-info">A curva ROC mostra o trade-off entre taxa de verdadeiro positivo (sensibilidade) '
            'e taxa de falso positivo (1-especificidade). AUC mais alto = melhor discriminacao.</div>',
            unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        try:
            st.plotly_chart(pr_chart(results), use_container_width=True, config={'displayModeBar': False})
        except Exception as exc:
            st.markdown(f'<div class="box-warning">Erro: {exc}</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="box-info">A curva Precisao-Recall e mais informativa que a ROC em datasets '
            'desbalanceados. AP (Average Precision) resume a area sob a curva. '
            'A linha tracejada representa o classificador aleatorio (prevalencia do dataset: 34.9%).</div>',
            unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        try:
            st.plotly_chart(calibration_chart(results), use_container_width=True, config={'displayModeBar': False})
        except Exception as exc:
            st.markdown(f'<div class="box-warning">Erro: {exc}</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="box-info">Um modelo bem calibrado deve ter sua curva proxima da diagonal. '
            'Curvas acima da diagonal indicam subestimacao; abaixo, superestimacao das probabilidades. '
            'Calibracao e essencial para uso clinico real das probabilidades previstas.</div>',
            unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab4:
        model_names = list(results.keys())
        cols = st.columns(min(2, len(model_names)))
        for i, name in enumerate(model_names):
            with cols[i % 2]:
                st.markdown(f'<div class="card">', unsafe_allow_html=True)
                st.markdown(f'<p class="section-title">{name}</p>', unsafe_allow_html=True)
                cm = results[name]['confusion_matrix']
                tn, fp, fn, tp = cm.ravel()
                try:
                    st.plotly_chart(
                        confusion_matrix_chart(cm, name),
                        use_container_width=True,
                        config={'displayModeBar': False},
                    )
                except Exception as exc:
                    st.markdown(f'<div class="box-warning">Erro: {exc}</div>', unsafe_allow_html=True)
                sensib = tp / (tp + fn) if (tp + fn) > 0 else 0
                espec = tn / (tn + fp) if (tn + fp) > 0 else 0
                mc1, mc2 = st.columns(2)
                with mc1:
                    st.markdown(f'<div class="kpi-card green" style="padding:0.7rem 1rem">'
                                f'<div class="kpi-value" style="font-size:1.2rem">{sensib:.3f}</div>'
                                f'<div class="kpi-label">Sensibilidade</div></div>', unsafe_allow_html=True)
                with mc2:
                    st.markdown(f'<div class="kpi-card blue" style="padding:0.7rem 1rem">'
                                f'<div class="kpi-value" style="font-size:1.2rem">{espec:.3f}</div>'
                                f'<div class="kpi-label">Especificidade</div></div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

    with tab5:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if cv_results:
            try:
                fig_cv = go.Figure()
                palette = [COLORS['teal'], COLORS['blue'], COLORS['amber'], COLORS['red']]
                for i, (name, cv) in enumerate(cv_results.items()):
                    scores = cv['auc_scores']
                    fig_cv.add_trace(go.Box(
                        y=scores, name=name,
                        marker_color=palette[i % len(palette)],
                        boxmean='sd',
                        hovertemplate='%{y:.4f}<extra></extra>',
                    ))
                fig_cv.update_layout(
                    **PLOT_TEMPLATE, height=380,
                    yaxis=dict(title='AUC (5-Fold CV)', range=[0.5, 1.0],
                               gridcolor='#F3F4F6', linecolor='#E2E8F0'),
                    xaxis=dict(gridcolor='#F3F4F6', linecolor='#E2E8F0', tickfont=dict(color='#1E293B', size=11)),
                    margin=dict(l=20, r=20, t=20, b=20),
                    showlegend=False,
                )
                st.plotly_chart(fig_cv, use_container_width=True, config={'displayModeBar': False})
            except Exception as exc:
                st.markdown(f'<div class="box-warning">Erro ao gerar grafico de validacao cruzada: {exc}</div>', unsafe_allow_html=True)

            header = "<thead><tr><th>Modelo</th><th>AUC Medio</th><th>Desvio Padrao</th><th>Acuracia Media</th></tr></thead>"
            rows = "".join(
                f"<tr><td><b>{name}</b></td>"
                f"<td>{cv['auc_mean']:.4f}</td>"
                f"<td>+/- {cv['auc_std']:.4f}</td>"
                f"<td>{cv['acc_mean']:.4f}</td></tr>"
                for name, cv in cv_results.items()
            )
            st.markdown(f'<table class="metrics-table"><{header}<tbody>{rows}</tbody></table>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="box-warning">Resultados de validacao cruzada nao disponiveis.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab6:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        try:
            st.plotly_chart(metrics_comparison_chart(results), use_container_width=True,
                            config={'displayModeBar': False})
        except Exception as exc:
            st.markdown(f'<div class="box-warning">Erro ao gerar grafico: {exc}</div>', unsafe_allow_html=True)

        feat_imp_models = {k: v for k, v in sd['models'].items()
                          if hasattr(v, 'feature_importances_')}
        if feat_imp_models:
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown('<p class="section-title">Importancia de Variaveis (Random Forest / Gradient Boosting)</p>',
                        unsafe_allow_html=True)
            fi_model_name = st.selectbox("Modelo", list(feat_imp_models.keys()), key='fi_sel')
            importances = feat_imp_models[fi_model_name].feature_importances_
            fi_df = pd.DataFrame({
                'feature': [FEATURE_LABELS[f] for f in FEATURE_NAMES],
                'importance': importances,
            }).sort_values('importance', ascending=True)
            try:
                fig_fi = go.Figure(go.Bar(
                    y=fi_df['feature'], x=fi_df['importance'], orientation='h',
                    marker_color=COLORS['teal'],
                    hovertemplate='%{y}: %{x:.4f}<extra></extra>',
                ))
                fig_fi.update_layout(
                    **PLOT_TEMPLATE, height=300,
                    xaxis=dict(title='Importancia (Gini)', gridcolor='#F3F4F6', linecolor='#E2E8F0'),
                    yaxis=dict(gridcolor='#F3F4F6', linecolor='#E2E8F0', tickfont=dict(color='#1E293B', size=11)),
                    margin=dict(l=20, r=20, t=20, b=20),
                )
                st.plotly_chart(fig_fi, use_container_width=True, config={'displayModeBar': False})
            except Exception as exc:
                st.markdown(f'<div class="box-warning">Erro ao gerar grafico: {exc}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


def page_ethics(sd: dict):
    df_clean = sd['df_clean']
    models = sd['models']
    X = sd['X']
    y = sd['y']
    scaler = sd['scaler']
    results = sd['results']

    st.markdown('<p class="page-title">Etica e Reflexao Critica</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Vieses, fairness, uso responsavel e limitacoes do sistema</p>',
                unsafe_allow_html=True)

    st.markdown(
        '<div class="box-warning" style="font-style:italic;font-size:0.9rem">'
        '"Um modelo de IA em saude que ignora seus vieses pode prejudicar exatamente as populacoes que pretende ajudar."'
        '</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Analise de Vieses", "Fairness por Grupo", "Uso Responsavel"])

    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">Importancia Global das Variaveis (SHAP Medio)</p>',
                    unsafe_allow_html=True)

        rf_model = next((v for k, v in models.items() if 'Forest' in k), None)
        gb_model = next((v for k, v in models.items() if 'Boosting' in k), None)
        ref_model = rf_model or gb_model

        if ref_model is not None and hasattr(ref_model, 'feature_importances_'):
            importances = ref_model.feature_importances_
            fi_df = pd.DataFrame({
                'feature': [FEATURE_LABELS[f] for f in FEATURE_NAMES],
                'importance': importances * 100,
            }).sort_values('importance', ascending=True)
            colors_bar = [COLORS['teal'] if i == len(fi_df) - 1 else '#60A5FA'
                          for i in range(len(fi_df))]
            try:
                fig = go.Figure(go.Bar(
                    y=fi_df['feature'], x=fi_df['importance'], orientation='h',
                    marker_color=colors_bar,
                    text=[f"{v:.1f}%" for v in fi_df['importance']],
                    textposition='outside',
                    hovertemplate='%{y}: %{x:.2f}%<extra></extra>',
                ))
                fig.update_layout(
                    **PLOT_TEMPLATE, height=310,
                    xaxis=dict(title='Importancia Relativa (%)', gridcolor='#F3F4F6', linecolor='#E2E8F0'),
                    yaxis=dict(gridcolor='#F3F4F6', linecolor='#E2E8F0', tickfont=dict(color='#1E293B', size=11)),
                    margin=dict(l=20, r=80, t=20, b=20),
                )
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            except Exception as exc:
                st.markdown(f'<div class="box-warning">Erro ao gerar grafico: {exc}</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<p class="section-title" style="margin-top:1rem">Vieses no Dataset</p>',
                        unsafe_allow_html=True)
            items = [
                ("Populacao restrita", "Apenas mulheres indigenas Pima (Arizona). Nao generaliza para outras populacoes."),
                ("Sub-representacao", "Grupos minoritarios podem ter desempenho inferior ao reportado."),
                ("Dados historicos", "Coleta nos anos 1980. Padroes de vida e diagnostico mudaram."),
                ("Zeros invalidos", f"{int((df_clean[['Glucose','BMI']]==0).sum().sum())} valores suspeitos imputados — pode introducir ruido."),
            ]
            for title, text in items:
                st.markdown(
                    f'<div style="margin:8px 0">'
                    f'<p style="font-weight:600;color:#1E293B;font-size:0.85rem;margin:0">{title}</p>'
                    f'<p style="color:#64748B;font-size:0.8rem;margin:2px 0 0">{text}</p>'
                    f'</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<p class="section-title" style="margin-top:1rem">Acoes de Mitigacao</p>',
                        unsafe_allow_html=True)
            actions = [
                ("Imputacao por classe", "Mediana calculada separadamente para diabeticos e nao-diabeticos."),
                ("Balanceamento de classes", "class_weight balanced nos modelos sklearn; pesos na rede neural."),
                ("Validacao cruzada", "5-Fold estratificado garante avaliacao robusta."),
                ("SHAP para transparencia", "Cada predicao acompanhada de explicacao das variaveis determinantes."),
                ("Limiar ajustavel", "Threshold de 0.5 pode ser ajustado conforme custo de falsos negativos."),
            ]
            for title, text in actions:
                st.markdown(
                    f'<div class="box-success" style="margin:6px 0">'
                    f'<strong>{title}:</strong> {text}'
                    f'</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">Desempenho por Faixa Etaria</p>', unsafe_allow_html=True)

        best_model_name = max(results, key=lambda n: results[n]['metrics']['ROC-AUC'])
        best_model = models[best_model_name]

        from src.evaluate import predict_proba as ep_proba

        df_eval = df_clean.copy()
        df_eval['Age_Group'] = pd.cut(
            df_eval['Age'],
            bins=[20, 30, 45, 60, 100],
            labels=['21-30', '31-45', '46-60', '60+'],
        )

        from src.preprocessing import preprocess
        X_all, y_all, _ = preprocess(sd['df'], scaler=scaler, fit_scaler=False)
        y_pred_all = (ep_proba(best_model, X_all) >= 0.5).astype(int)
        df_eval['y_pred'] = y_pred_all
        df_eval['correct'] = (df_eval['y_pred'] == df_eval['Outcome']).astype(int)

        age_stats = df_eval.groupby('Age_Group', observed=True).agg(
            n=('Outcome', 'count'),
            prevalencia=('Outcome', 'mean'),
            acuracia=('correct', 'mean'),
        ).reset_index()

        try:
            fig_age = make_subplots(specs=[[{"secondary_y": True}]])
            fig_age.add_trace(go.Bar(
                x=age_stats['Age_Group'].astype(str), y=age_stats['acuracia'],
                name='Acuracia', marker_color=COLORS['teal'], opacity=0.85,
            ), secondary_y=False)
            fig_age.add_trace(go.Scatter(
                x=age_stats['Age_Group'].astype(str), y=age_stats['prevalencia'],
                name='Prevalencia de Diabetes', mode='lines+markers',
                line=dict(color=COLORS['red'], width=2.5),
                marker=dict(size=8),
            ), secondary_y=True)
            fig_age.update_layout(
                **PLOT_TEMPLATE, height=340,
                legend=dict(x=0.5, y=1.08, xanchor='center', orientation='h'),
                margin=dict(l=20, r=20, t=20, b=20),
            )
            fig_age.update_yaxes(title_text='Acuracia do Modelo', secondary_y=False,
                                 range=[0.5, 1.0], gridcolor='#F3F4F6')
            fig_age.update_yaxes(title_text='Prevalencia', secondary_y=True,
                                 range=[0, 1.0], gridcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_age, use_container_width=True, config={'displayModeBar': False})
        except Exception as exc:
            st.markdown(f'<div class="box-warning">Erro ao gerar grafico por faixa etaria: {exc}</div>', unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<p class="section-title">Desempenho por Categoria de IMC</p>',
                    unsafe_allow_html=True)
        df_eval['BMI_Group'] = pd.cut(
            df_eval['BMI'],
            bins=[0, 18.5, 25, 30, 200],
            labels=['Abaixo do peso', 'Normal', 'Sobrepeso', 'Obeso'],
        )
        bmi_stats = df_eval.groupby('BMI_Group', observed=True).agg(
            n=('Outcome', 'count'),
            prevalencia=('Outcome', 'mean'),
            acuracia=('correct', 'mean'),
        ).reset_index()

        try:
            fig_bmi = go.Figure()
            fig_bmi.add_trace(go.Bar(
                x=bmi_stats['BMI_Group'].astype(str), y=bmi_stats['acuracia'],
                name='Acuracia', marker_color=COLORS['blue'],
            ))
            fig_bmi.add_trace(go.Scatter(
                x=bmi_stats['BMI_Group'].astype(str), y=bmi_stats['prevalencia'],
                name='Prevalencia', mode='lines+markers',
                line=dict(color=COLORS['red'], width=2.5), marker=dict(size=8),
                yaxis='y2',
            ))
            fig_bmi.update_layout(
                **PLOT_TEMPLATE, height=300,
                yaxis=dict(title='Acuracia', range=[0.5, 1.0], gridcolor='#F3F4F6', linecolor='#E2E8F0', tickfont=dict(color='#1E293B', size=11)),
                yaxis2=dict(title='Prevalencia', overlaying='y', side='right', range=[0, 1]),
                legend=dict(x=0.5, y=1.08, xanchor='center', orientation='h'),
                margin=dict(l=20, r=60, t=20, b=20),
            )
            st.plotly_chart(fig_bmi, use_container_width=True, config={'displayModeBar': False})
        except Exception as exc:
            st.markdown(f'<div class="box-warning">Erro ao gerar grafico por IMC: {exc}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<p class="section-title">Uso Responsavel</p>', unsafe_allow_html=True)
            items_resp = [
                ("Suporte, nao substituicao",
                 "O sistema apoia o raciocinio clinico. A decisao final e sempre do profissional de saude."),
                ("Contexto necessario",
                 "Dados laboratoriais adicionais, historico familiar e exame fisico sao insubstituiveis."),
                ("Revisa periodicamente",
                 "Modelos treinados em dados historicos podem degradar com mudancas populacionais."),
                ("Privacidade",
                 "Dados de saude sao sensiveis. Implementar criptografia e controle de acesso em producao."),
            ]
            for title, text in items_resp:
                st.markdown(
                    f'<div style="margin:12px 0;padding-bottom:10px;border-bottom:1px solid #F1F5F9">'
                    f'<p style="font-weight:700;color:#0D9488;font-size:0.85rem;margin:0">{title}</p>'
                    f'<p style="color:#475569;font-size:0.82rem;margin:4px 0 0">{text}</p>'
                    f'</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<p class="section-title">Transparencia do Modelo</p>', unsafe_allow_html=True)
            st.markdown(
                '<div class="box-info">Este sistema implementa explicabilidade via SHAP (SHapley Additive '
                'exPlanations), permitindo que cada predicao seja acompanhada de uma explicacao das '
                'variaveis mais determinantes.</div>', unsafe_allow_html=True)
            st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
            items_transp = [
                ("Rastreabilidade", "Todos os modelos e parametros estao versionados no repositorio."),
                ("Reproducibilidade", "random_state=42 em todos os algoritmos garante resultados identicos."),
                ("Auditoria", "Pipeline documentado em Jupyter Notebook com EDA detalhada."),
                ("Limiar ajustavel", "Threshold de classificacao configuravel para equilibrar sensibilidade e especificidade."),
            ]
            for title, text in items_transp:
                st.markdown(
                    f'<div style="margin:10px 0">'
                    f'<p style="font-weight:600;color:#1E293B;font-size:0.85rem;margin:0">{title}</p>'
                    f'<p style="color:#64748B;font-size:0.8rem;margin:2px 0 0">{text}</p>'
                    f'</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">Limitacoes Conhecidas e Recomendacoes</p>',
                    unsafe_allow_html=True)
        limits = [
            ("Dataset restrito", "Expandir para datasets mais diversos (NHANES, DATASUS) para generalizar o modelo."),
            ("Variaveis ausentes", "Incluir HbA1c, historico familiar detalhado e dados socioeconomicos."),
            ("Desequilibrio de classes", "Explorar tecnicas SMOTE ou cost-sensitive learning para melhorar recall em diabeticos."),
            ("Calibracao", "Calibrar probabilidades com Platt Scaling ou Isotonic Regression para uso clinico real."),
            ("Prospecto de validacao", "Validar o modelo em dados prospectivos coletados apos o treinamento."),
        ]
        ll, lr = st.columns(2)
        for i, (title, text) in enumerate(limits):
            with (ll if i % 2 == 0 else lr):
                st.markdown(
                    f'<div style="margin:8px 0;padding:10px 14px;background:#FFF7ED;'
                    f'border-left:3px solid {COLORS["amber"]};border-radius:0 8px 8px 0">'
                    f'<p style="font-weight:600;color:#92400E;font-size:0.82rem;margin:0">{title}</p>'
                    f'<p style="color:#78350F;font-size:0.79rem;margin:3px 0 0">{text}</p>'
                    f'</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    try:
        sd = init_system()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"Erro ao inicializar o sistema: {e}")
        st.stop()

    best_name = max(sd['results'], key=lambda n: sd['results'][n]['metrics']['ROC-AUC'])
    best_auc = sd['results'][best_name]['metrics']['ROC-AUC']

    with st.sidebar:
        st.markdown(
            '<p class="brand">PimaGuard</p>'
            '<p class="brand-sub">Diabetes Tipo 2</p>',
            unsafe_allow_html=True)

        st.markdown('<div style="height:0.8rem"></div>', unsafe_allow_html=True)

        page = st.radio(
            "nav",
            ["Dashboard", "Predicao de Risco", "Analise de Dados",
             "Comparacao de Modelos", "Etica e Transparencia"],
            label_visibility="collapsed",
        )

        st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="sidebar-meta">'
            f'<p style="color:#94A3B8 !important;font-size:0.72rem;margin:0 0 4px">DATASET</p>'
            f'<p style="color:#CBD5E1 !important;font-weight:600;margin:0">Pima Indians</p>'
            f'<p style="margin:2px 0">768 pacientes | 8 variaveis</p>'
            f'<p style="margin:0">34.9% diabeticos</p>'
            f'</div>',
            unsafe_allow_html=True)
        st.markdown(
            f'<div class="sidebar-meta">'
            f'<p style="color:#94A3B8 !important;font-size:0.72rem;margin:0 0 4px">MELHOR MODELO</p>'
            f'<p style="color:#CBD5E1 !important;font-weight:600;margin:0">{best_name}</p>'
            f'<p style="margin:2px 0">AUC: {best_auc:.4f}</p>'
            f'</div>',
            unsafe_allow_html=True)
        st.markdown(
            '<div class="sidebar-meta">'
            '<p style="color:#94A3B8 !important;font-size:0.72rem;margin:0 0 4px">ALGORITMOS</p>'
            + "".join(
                f'<p style="margin:2px 0;color:#8DA2BC !important">{n}</p>'
                for n in sd['models']
            )
            + '</div>',
            unsafe_allow_html=True)

        st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)

    pages = {
        "Dashboard": page_dashboard,
        "Predicao de Risco": page_prediction,
        "Analise de Dados": page_eda,
        "Comparacao de Modelos": page_models,
        "Etica e Transparencia": page_ethics,
    }
    pages[page](sd)


if __name__ == '__main__':
    main()
