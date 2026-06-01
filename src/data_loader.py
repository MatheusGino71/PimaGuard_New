import os
import pandas as pd
import numpy as np

FEATURE_NAMES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age',
]
TARGET = 'Outcome'

FEATURE_LABELS = {
    'Pregnancies': 'Gestacoes',
    'Glucose': 'Glicose',
    'BloodPressure': 'Pressao Arterial',
    'SkinThickness': 'Espessura da Pele',
    'Insulin': 'Insulina',
    'BMI': 'IMC',
    'DiabetesPedigreeFunction': 'Pedigree Diabetes',
    'Age': 'Idade',
}

FEATURE_UNITS = {
    'Pregnancies': '',
    'Glucose': 'mg/dL',
    'BloodPressure': 'mmHg',
    'SkinThickness': 'mm',
    'Insulin': 'uU/mL',
    'BMI': 'kg/m2',
    'DiabetesPedigreeFunction': '',
    'Age': 'anos',
}

ZERO_INVALID_COLS = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'diabetes.csv')


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path)

    try:
        import kagglehub
        print("Baixando dataset do Kaggle...")

        # kagglehub >= 0.3: usa KaggleDatasetAdapter
        try:
            from kagglehub import KaggleDatasetAdapter
            df = kagglehub.load_dataset(
                KaggleDatasetAdapter.PANDAS,
                "uciml/pima-indians-diabetes-database",
                "",
            )
        except (ImportError, AttributeError):
            # kagglehub < 0.3: baixa arquivos e le o CSV manualmente
            import glob
            dataset_path = kagglehub.dataset_download("uciml/pima-indians-diabetes-database")
            csv_files = glob.glob(os.path.join(dataset_path, "*.csv"))
            if not csv_files:
                raise FileNotFoundError(f"Nenhum CSV encontrado em {dataset_path}")
            df = pd.read_csv(csv_files[0])

        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        print(f"Dataset salvo em: {path}")
        return df
    except Exception as e:
        raise FileNotFoundError(
            f"Dataset nao encontrado em '{path}'.\n"
            "Opcoes:\n"
            "  1. Baixe o CSV em https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database\n"
            "     e salve em 'data/diabetes.csv'.\n"
            "  2. Configure o kagglehub com suas credenciais do Kaggle.\n"
            f"Erro original: {e}"
        )
