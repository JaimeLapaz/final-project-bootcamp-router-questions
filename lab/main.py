import os
from preprocess_data import preprocess
from model_train import config
import mlflow
from evaluation import evaluation_config


def main():
    # Configuración de MLflow
    mlflow.set_tracking_uri("http://localhost:5050")
    mlflow.set_experiment("enrutador_de_preguntas")
    
    # Preprocesamiento de los dataset
    preprocess.preprocess_dataset()
    model_config_path = os.path.join(config.PATH_MY_MODEL, 'config.json')
    
    if not os.path.exists(model_config_path):
        # Entrenar modelo si no existe
        data = preprocess.load_data("data_train.json")
        questions, labels, label_encoder = preprocess.preprocess_data(data)
        train_questions, val_questions, train_labels, val_labels = preprocess.split_data(questions, labels)

        # Entrenar modelo
        config.train_model(train_questions, val_questions, train_labels, val_labels, label_encoder)
        print("Entrenamiento del modelo completado.")
    
    # Evaluar modelo
    try:
        evaluation_config.evaluation_model()
    except Exception as e:
        print(f"Error durante la evaluación del modelo: {e}")

if __name__ == "__main__":
    main()