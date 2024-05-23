from transformers import BertTokenizer, BertForSequenceClassification
import json
import uuid
import joblib
import sys
import os
import re
from evaluation.data_processing_test import get_data_text_xml_to_json
from model_train import config
from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.getenv('PYTHONPATH'))


# Función para limpiar el texto
def clean_text(text: str) -> str:
        # Caracteres especiales de UFT-8
        text = text.encode('utf-8').decode('unicode_escape')
        # Texto a minuscula
        text = text.lower()
        # Eliminar caracteres especiales y números
        text = re.sub(r'[^a-z]', ' ', text)
        return text

    # Función para clasificar la pregunta
def classify_question(question, tokenizer, model, label_encoder):
    question_clean = clean_text(question)
    inputs = tokenizer(question_clean, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    predicted_class_idx = outputs.logits.argmax(dim=-1).item()
    predicted_class = label_encoder.inverse_transform([predicted_class_idx])[0]
    return predicted_class

# Cargar la base de datos de respuestas desde un archivo JSON
def load_responses(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        responses = json.load(file)
    return responses


# Función para obtener la respuesta adecuada basada en la clasificación
def get_response(question, classification, responses_db):
    answers = []
    type_original = None
    for d in responses_db:
        if d['focus'].lower() in question.lower():
            type_original = d['type_info']
            if d['type_info'] == classification:
                for answer in d['answer_list']:
                    answers.append(answer['text'].encode('utf-8').decode('unicode_escape'))
    if not answers:
        answers.append('There is no information available for that question.')
    return answers, type_original

    # Función para registrar las interacciones en un archivo JSON
def log_interaction(log_file, question, question_id, classification, response, type_original):
    log_entry = {
            "id": str(uuid.uuid4()),
            "question_id":question_id,
            "question": question,
            "classification": classification,
            "original_type": type_original,
            "response": response
        }
    try:
        with open(log_file, 'r+', encoding='utf-8') as file:
            logs = json.load(file)
            logs.append(log_entry)
            file.seek(0)
            json.dump(logs, file, ensure_ascii=False, indent=4)
    except FileNotFoundError:
        with open(log_file, 'w', encoding='utf-8') as file:
            json.dump([log_entry], file, ensure_ascii=False, indent=4)

    # Función principal para manejar la pregunta
def handle_question(responses_db, log_file, tokenizer, model, label_encoder):
    aciertos = 0 
    for data in responses_db:
        question = data['question_text'].encode('utf-8').decode('unicode_escape')
        question_test = question + data['nist_paraphrase'] + data['nlm_summary'] + data['subject']
        question_test = question_test.encode('utf-8').decode('unicode_escape')
        question_id = data['question_id']
        classification = classify_question(question_test, tokenizer, model, label_encoder)
        response,  type_original = get_response(question_test, classification, responses_db)
        log_interaction(log_file, question, question_id, classification, response, type_original)
        if classification == type_original:
            aciertos += 1
    return response,aciertos


def evaluation_model():
    # Creamos el archivo json
    get_data_text_xml_to_json()
    
    
    model_path = f"{os.getenv('PYTHONPATH')}/lab/my_model/{config.NAME_MODEL_SAVE}"
    label_encoder_path = f"{os.getenv('PYTHONPATH')}/lab/my_model/label_encoder.pkl"    
    responses_db_path = f"{os.getenv('PYTHONPATH')}/lab/evaluation/Data/data_test.json"
    log_file = f"{os.getenv('PYTHONPATH')}/lab/evaluation/registro.json"
    
    
    if not os.path.exists(model_path):
        print(f"El modelo no existe en la ruta. Por favor, entrena el modelo antes de la evaluación.")
        return
    
    print(f"Modelo encontrado. Cargando modelo...")  # Mensaje de depuración
    
    responses_db = load_responses(responses_db_path)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    label_encoder = joblib.load(label_encoder_path)
    response, aciertos = handle_question(responses_db, log_file, tokenizer, model, label_encoder)
    porcent_aciertos = round((aciertos/len(responses_db))*100, 2)
    
    print(f"Numero de aciertos: {aciertos} \nPorcentaje de aciertos {porcent_aciertos}%")