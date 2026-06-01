#!/usr/bin/env python3
"""
PimaGuard - Pipeline de Treinamento Completo
Executa uma vez para treinar e salvar todos os modelos (incluindo Keras).

Uso:
    python train_pipeline.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import joblib
from src.data_loader import load_data, FEATURE_NAMES
from src.preprocessing import preprocess, split_data
from src.train import train_sklearn, train_keras, MODELS_DIR
from src.evaluate import evaluate_all


def main():
    print("=" * 55)
    print("  PimaGuard - Pipeline de Treinamento")
    print("  Diabetes Tipo 2 | Pima Indians Dataset")
    print("=" * 55)

    print("\n[1/5] Carregando dataset...")
    df = load_data()
    neg = (df['Outcome'] == 0).sum()
    pos = (df['Outcome'] == 1).sum()
    print(f"      Total: {len(df)} pacientes")
    print(f"      Nao-diabetico: {neg} ({neg/len(df)*100:.1f}%)")
    print(f"      Diabetico:     {pos} ({pos/len(df)*100:.1f}%)")

    print("\n[2/5] Pre-processando...")
    X, y, scaler = preprocess(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"      Treino: {X_train.shape[0]}  |  Teste: {X_test.shape[0]}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    print("      Scaler salvo.")

    print("\n[3/5] Treinando modelos sklearn...")
    sklearn_models = train_sklearn(X_train, y_train)

    print("\n[4/5] Treinando Rede Neural Keras (pode demorar)...")
    keras_model = train_keras(X_train, y_train)

    all_models = dict(sklearn_models)
    if keras_model is not None:
        all_models['Rede Neural Keras'] = keras_model

    print("\n[5/5] Avaliando modelos no conjunto de teste...")
    results = evaluate_all(all_models, X_test, y_test)

    print("\n" + "=" * 55)
    print(f"  {'Modelo':<22} {'Acuracia':>9} {'F1':>9} {'AUC':>9}")
    print("  " + "-" * 51)
    for name, data in results.items():
        m = data['metrics']
        print(f"  {name:<22} {m['Acuracia']:>9.4f} {m['F1-Score']:>9.4f} {m['ROC-AUC']:>9.4f}")
    print("=" * 55)

    print("\nTreinamento concluido.")
    print("Execute: streamlit run app.py\n")


if __name__ == '__main__':
    main()
