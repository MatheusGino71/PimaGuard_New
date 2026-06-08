# PimaGuard — Sistema de Avaliacao de Risco para Diabetes Tipo 2

> Prototipo de IA clinica com explicabilidade (SHAP), multiplos algoritmos de ML e dashboard interativo.

---

## Visao Geral

**PimaGuard** e uma aplicacao web de apoio a decisao clinica que estima o risco de Diabetes Tipo 2 com base em dados biometricos e laboratoriais. O sistema treina e compara 4 algoritmos de Machine Learning, oferece explicacoes individuais por paciente via SHAP e apresenta analises de fairness por grupo demografico.

O projeto foi desenvolvido como trabalho pratico da disciplina **Inteligencia Artificial Aplicada a Saude**, com enfase em uso responsavel de IA, transparencia algorítmica e acessibilidade clinica.

---

## Demo

Acesse a aplicacao em producao:

https://pimaguard.streamlit.app/
---

## Funcionalidades

| Pagina | Descricao |
|--------|-----------|
| **Dashboard** | KPIs globais, comparacao de modelos, distribuicao do dataset |
| **Predicao de Risco** | Formulario clinico interativo, gauge de probabilidade, explicacao SHAP por paciente |
| **Analise de Dados** | Distribuicoes, correlacoes, box plots comparativos, dispersao |
| **Comparacao de Modelos** | Curvas ROC, matrizes de confusao, validacao cruzada 5-Fold |
| **Etica e Transparencia** | Analise de vieses, fairness por faixa etaria/IMC, uso responsavel |

---

## Modelos Treinados

| Modelo | AUC-ROC | Acuracia | Recall |
|--------|---------|----------|--------|
| Gradient Boosting | ~0.958 | ~88% | ~82% |
| Random Forest | ~0.940 | ~85% | ~78% |
| Rede Neural MLP | ~0.920 | ~83% | ~78% |
| Regressao Logistica | ~0.840 | ~77% | ~72% |

> Valores aproximados — re-treinamento com `random_state=42` produz resultados identicos.

---

## Stack Tecnologica

```
Frontend    →  Streamlit 1.35+
ML          →  scikit-learn 1.3+ (LR, RF, GB, MLP)
Deep Learn  →  TensorFlow / Keras 2.13+ (opcional)
Explicab.   →  SHAP 0.44+
Visualiz.   →  Plotly 5.18+, Matplotlib, Seaborn
Dados       →  Pandas 2.0+, NumPy 1.24+
Dataset     →  Pima Indians Diabetes Database (Kaggle / UCI)
```

---

## Dataset

**Pima Indians Diabetes Database** — 768 pacientes do sexo feminino da populacao indigena Pima (Arizona, EUA).

| Variavel | Tipo | Descricao |
|----------|------|-----------|
| Pregnancies | int | Numero de gestacoes |
| Glucose | float | Glicose plasmatica 2h apos TOTG (mg/dL) |
| BloodPressure | float | Pressao arterial diastolica (mmHg) |
| SkinThickness | float | Espessura da dobra cutanea do triceps (mm) |
| Insulin | float | Insulina serica 2h apos TOTG (uU/mL) |
| BMI | float | Indice de Massa Corporal (kg/m²) |
| DiabetesPedigreeFunction | float | Funcao de risco genetico familiar |
| Age | int | Idade (anos) |
| **Outcome** | int | **Alvo**: 1 = diabetico, 0 = nao-diabetico |

**Distribuicao:** 268 diabeticos (34.9%) / 500 nao-diabeticos (65.1%)

---

## Como Executar Localmente

### Pre-requisitos

- Python 3.8 ou superior
- pip

### Instalacao

```bash
# 1. Clone o repositorio
git clone https://github.com/MatheusGino71/PimaGuard_New.git
cd PimaGuard_New

# 2. Crie e ative um ambiente virtual (opcional, recomendado)
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Instale as dependencias
pip install -r requirements.txt
```

### Execucao

```bash
# Os modelos ja estao pre-treinados na pasta models/
# Apenas inicie o app:
streamlit run app.py
```

Acesse `http://localhost:8501` no browser.

### Re-treinar os modelos (opcional)

```bash
python train_pipeline.py
```

---

## Estrutura do Projeto

```
PimaGuard_New/
├── app.py                   # Aplicacao Streamlit principal
├── train_pipeline.py        # Pipeline de treinamento standalone
├── requirements.txt         # Dependencias Python
│
├── src/
│   ├── data_loader.py       # Carregamento e definicao de features
│   ├── preprocessing.py     # Limpeza, imputacao e normalizacao
│   ├── train.py             # Treinamento dos 4 algoritmos
│   └── evaluate.py          # Metricas, SHAP e validacao cruzada
│
├── data/
│   └── diabetes.csv         # Dataset Pima Indians
│
├── models/
│   ├── Regressao_Logistica.pkl
│   ├── Random_Forest.pkl
│   ├── Gradient_Boosting.pkl
│   ├── Rede_Neural_MLP.pkl
│   └── scaler.pkl
│
└── notebooks/
    └── 01_eda_modelagem.ipynb   # EDA detalhada e experimentos
```

---

## Deploy no Streamlit Community Cloud

1. Faca fork ou clone deste repositorio para sua conta GitHub
2. Acesse [share.streamlit.io](https://share.streamlit.io)
3. Clique em **New app** e selecione este repositorio
4. Configure:
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Clique em **Deploy**

> O Streamlit Cloud instala as dependencias do `requirements.txt` automaticamente.
> Os modelos pre-treinados estao incluidos no repositorio — nao e necessario re-treinar.

---

## Pre-processamento

- **Zeros invalidos:** substituicao por NaN nas colunas Glicose, Pressao, Espessura, Insulina e IMC
- **Imputacao:** mediana calculada separadamente para diabeticos e nao-diabeticos (evita vazamento de informacao)
- **Normalizacao:** StandardScaler (media=0, desvio=1)
- **Divisao:** 80% treino / 20% teste, estratificada por classe

---

## Consideracoes Eticas

- **Populacao restrita:** dataset coletado exclusivamente de mulheres indigenas Pima nos anos 1980 — nao deve ser generalizado sem validacao externa
- **Balanceamento:** `class_weight='balanced'` em todos os modelos sklearn
- **Explicabilidade:** SHAP para cada predicao individual
- **Uso clinico:** este sistema e um prototipo de apoio a decisao — **nao substitui avaliacao medica profissional**

---

## Licenca

Projeto academico de uso livre para fins educacionais.
