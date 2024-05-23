import os
import sys
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from flask import Flask, request, jsonify, render_template
import json
import uuid
import joblib
from dotenv import load_dotenv
import time

load_dotenv()

# Agregar verificación de variables de entorno
PATH_PROJECT = os.getenv('PYTHONPATH')
if not PATH_PROJECT:
    raise ValueError("PYTHONPATH environment variable not set")

# Configuraciones del modelo y archivos
MODEL_NAME = "bert-base-uncased"
BATCH_SIZE_MODEL = 16
EPOCH_SIZE_MODEL = 5
NAME_MODEL_SAVE = f"{MODEL_NAME}_{BATCH_SIZE_MODEL}_{EPOCH_SIZE_MODEL}"
MODEL_PATH = os.path.join(PATH_PROJECT, f"my_project/model/{NAME_MODEL_SAVE}")
LABEL_ENCODER_PATH = os.path.join(PATH_PROJECT, 'my_project/model/label_encoder.pkl')
RESPONSES_DB_PATH = os.path.join(PATH_PROJECT, 'my_project/responses.json')
LOG_FILE = os.path.join(PATH_PROJECT, 'my_project/registro.json')

# Inicializar Flask
app = Flask(__name__)

# Cargar el modelo y el tokenizer con control de errores
try:
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    sys.exit(1)

# Cargar el LabelEncoder con control de errores
try:
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
except Exception as e:
    print(f"Error loading label encoder: {e}")
    sys.exit(1)

# Función para clasificar la pregunta
def classify_question(question):
    inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class_idx = torch.argmax(probabilities, dim=-1).item()
    predicted_class = label_encoder.inverse_transform([predicted_class_idx])[0]
    return predicted_class

# Cargar la base de datos de respuestas desde un archivo JSON
def load_responses(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            responses = json.load(file)
    except FileNotFoundError:
        print(f"Responses database file not found at {file_path}")
        responses = []
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {file_path}: {e}")
        responses = []
    return responses

responses_db = load_responses(RESPONSES_DB_PATH)

# Función para obtener la respuesta adecuada basada en la clasificación
def get_response(question, responses_db):
    answers = []
    type_original = None
    for d in responses_db:
        if d['focus'].lower() in question.lower():
            type_original = d['type_info']
            for answer in d['answer_list']:
                answers.append(answer['text'].encode('utf-8').decode('unicode_escape'))
    if not answers:
        answers.append('There is no information available for that question.')
    return answers, type_original

# Función para registrar las interacciones en un archivo JSON
def log_interaction(log_file, question, classification, response, type_info_original, response_time):
    log_entry = {
        "id": str(uuid.uuid4()),
        "question": question,
        "classification": classification,
        "type_info_original": type_info_original,
        "response": response,
        "response_time": response_time,
        "useful": None
    }
    try:
        with open(log_file, 'r+', encoding='utf-8') as file:
            try:
                logs = json.load(file)
            except json.JSONDecodeError:
                logs = []
            logs.append(log_entry)
            file.seek(0)
            json.dump(logs, file, ensure_ascii=False, indent=4)
    except FileNotFoundError:
        with open(log_file, 'w', encoding='utf-8') as file:
            json.dump([log_entry], file, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error logging interaction: {e}")

# Función para actualizar la utilidad en el registro
def update_usefulness(log_file, interaction_id, useful):
    try:
        with open(log_file, 'r+', encoding='utf-8') as file:
            logs = json.load(file)
            for entry in logs:
                if entry["id"] == interaction_id:
                    entry["useful"] = useful
                    break
            file.seek(0)
            json.dump(logs, file, ensure_ascii=False, indent=4)
    except FileNotFoundError:
        print(f"Log file {log_file} not found.")
    except Exception as e:
        print(f"Error updating usefulness: {e}")

# Rutas de la aplicación Flask
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question')
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    start_time = time.time()
    classification = classify_question(question)
    response, type_original = get_response(question, responses_db)
    end_time = time.time()

    response_time = end_time - start_time
    interaction_id = str(uuid.uuid4())
    log_interaction(LOG_FILE, question, classification, response, type_original, response_time)
    
    return jsonify({
        'id': interaction_id,
        'question': question,
        'classification': classification,
        'responses': response
    })

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.json
    interaction_id = data.get('id')
    useful = data.get('useful')
    update_usefulness(LOG_FILE, interaction_id, useful)
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True)
