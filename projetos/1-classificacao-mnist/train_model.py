import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


SEED = 42
BATCH_SIZE = 128
MAX_EPOCHS = 15


def carregar_dados():
    """Carrega, normaliza e separa os dados de treino e validação."""
    (x_train, y_train), _ = keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, axis=-1)

    # Embaralhamento reproduzível antes da separação.
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(x_train))
    x_train = x_train[indices]
    y_train = y_train[indices]

    # 55.000 imagens para treino e 5.000 para validação.
    x_val = x_train[-5000:]
    y_val = y_train[-5000:]
    x_train = x_train[:-5000]
    y_train = y_train[:-5000]

    return x_train, y_train, x_val, y_val


def construir_modelo():
    """Constrói uma CNN simples para classificação dos dígitos."""
    model = keras.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),

            layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main():
    tf.keras.utils.set_random_seed(SEED)

    x_train, y_train, x_val, y_val = carregar_dados()
    model = construir_modelo()

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    )

    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=2,
    )

    val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose=0)
    print(f"\nPerda final de validacao: {val_loss:.4f}")
    print(f"Acuracia final de validacao: {val_accuracy:.2%}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model.h5")
    model.save(model_path)

    print(f"Modelo salvo em: {model_path}")


if __name__ == "__main__":
    main()