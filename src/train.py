import os
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

MODEL_FILES = {
    'Regressao Logistica': 'Regressao_Logistica.pkl',
    'Random Forest': 'Random_Forest.pkl',
    'Gradient Boosting': 'Gradient_Boosting.pkl',
    'Rede Neural MLP': 'Rede_Neural_MLP.pkl',
    'Rede Neural Keras': 'Rede_Neural_Keras.keras',
}


def _sklearn_configs():
    return {
        'Regressao Logistica': LogisticRegression(
            C=0.5, max_iter=2000, random_state=42, class_weight='balanced',
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_split=4,
            random_state=42, class_weight='balanced', n_jobs=-1,
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=250, learning_rate=0.05, max_depth=4,
            subsample=0.8, random_state=42,
        ),
        'Rede Neural MLP': MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), activation='relu',
            alpha=0.01, max_iter=800, random_state=42,
            early_stopping=True, validation_fraction=0.1,
            learning_rate_init=0.001, batch_size=32,
        ),
    }


def build_keras_model(input_dim: int):
    try:
        import tensorflow as tf
        from tensorflow import keras

        tf.random.set_seed(42)
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.25),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation='sigmoid'),
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')],
        )
        return model
    except ImportError:
        return None


def train_sklearn(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    os.makedirs(MODELS_DIR, exist_ok=True)
    models = {}
    for name, model in _sklearn_configs().items():
        print(f"  Treinando: {name}...")
        model.fit(X_train, y_train)
        fname = MODEL_FILES[name]
        joblib.dump(model, os.path.join(MODELS_DIR, fname))
        models[name] = model
    return models


def train_keras(X_train: np.ndarray, y_train: np.ndarray) -> object:
    os.makedirs(MODELS_DIR, exist_ok=True)
    model = build_keras_model(X_train.shape[1])
    if model is None:
        print("  TensorFlow nao disponivel, pulando Rede Neural Keras.")
        return None

    try:
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        callbacks = [
            EarlyStopping(patience=25, restore_best_weights=True, monitor='val_auc', mode='max'),
            ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=10, mode='max'),
        ]
        print("  Treinando: Rede Neural Keras...")
        model.fit(
            X_train, y_train,
            validation_split=0.15,
            epochs=300,
            batch_size=32,
            callbacks=callbacks,
            class_weight={0: 1.0, 1: 1.8},
            verbose=0,
        )
        model.save(os.path.join(MODELS_DIR, MODEL_FILES['Rede Neural Keras']))
        return model
    except Exception as e:
        print(f"  Erro ao treinar Keras: {e}")
        return None


def load_models() -> dict:
    models = {}
    if not os.path.exists(MODELS_DIR):
        return models

    for name, fname in MODEL_FILES.items():
        path = os.path.join(MODELS_DIR, fname)
        if not os.path.exists(path):
            continue
        if fname.endswith('.pkl'):
            try:
                models[name] = joblib.load(path)
            except Exception:
                pass
        elif fname.endswith('.keras'):
            try:
                import tensorflow as tf
                models[name] = tf.keras.models.load_model(path)
            except Exception:
                pass

    return models


def models_need_training() -> bool:
    required = ['Regressao_Logistica.pkl', 'Random_Forest.pkl', 'Gradient_Boosting.pkl', 'Rede_Neural_MLP.pkl']
    return not all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in required)
