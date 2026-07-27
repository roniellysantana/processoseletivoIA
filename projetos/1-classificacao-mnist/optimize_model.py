import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    h5_path = os.path.join(script_dir, "model.h5")
    tflite_path = os.path.join(script_dir, "model.tflite")

    if not os.path.exists(h5_path):
        raise FileNotFoundError(
            "Arquivo model.h5 nao encontrado. Execute train_model.py primeiro."
        )

    print(f"Carregando modelo: {h5_path}")
    model = tf.keras.models.load_model(h5_path, compile=False)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Dynamic Range Quantization para reduzir o tamanho do modelo.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    print("Convertendo e quantizando o modelo...")
    tflite_model = converter.convert()

    with open(tflite_path, "wb") as file:
        file.write(tflite_model)

    h5_size = os.path.getsize(h5_path) / (1024 * 1024)
    tflite_size = os.path.getsize(tflite_path) / (1024 * 1024)
    reduction = (1 - (tflite_size / h5_size)) * 100

    print(f"\nModelo original: {h5_size:.2f} MB")
    print(f"Modelo otimizado: {tflite_size:.2f} MB")
    print(f"Reducao de tamanho: {reduction:.2f}%")
    print(f"Modelo TFLite salvo em: {tflite_path}")


if __name__ == "__main__":
    main()