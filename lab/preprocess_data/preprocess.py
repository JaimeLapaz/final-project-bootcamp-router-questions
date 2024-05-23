
import os
import json
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from dotenv import load_dotenv
import sys
import mlflow
# Cargar variables de entorno desde .env
load_dotenv()

# Agregar PYTHONPATH al sys.path
sys.path.append(os.getenv("PYTHONPATH"))

#Cargamos las funciones para generar los documentos
from preprocess_data.dataset_processing import combine_datasets_train, get_data_train_xml_to_json, RUTE_NEW_DATASET

# Nombre de los archivos y nombre del archivo nuevo
DATASETS_1 = "TREC-2017-LiveQA-Medical-Train-1.xml"
DATASETS_2 = "TREC-2017-LiveQA-Medical-Train-2.xml"
NEW_DATASET = "new_dataset.xml"


def preprocess_dataset():
    combine_datasets_train(DATASETS_1, DATASETS_2, NEW_DATASET)
    get_data_train_xml_to_json(NEW_DATASET)

# Función para limpiar el texto
def clean_text(text: str) -> str:
    # Caracteres especiales de UFT-8
    text = text.encode('utf-8').decode('unicode_escape')
    # Texto a minuscula
    text = text.lower()
    # Eliminar caracteres especiales y números
    text = re.sub(r'[^a-z]', ' ', text)
    return text

def load_data(file_path):
    with open(RUTE_NEW_DATASET + file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def preprocess_data(data):
    questions = [d["question_text"] for d in data]
    questions_clean = [clean_text(d) for d in questions]
    labels = [d["type_info"] for d in data]
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    return questions_clean, labels, label_encoder

def split_data(questions, labels):
    return train_test_split(questions, labels, test_size=0.2, random_state=42)