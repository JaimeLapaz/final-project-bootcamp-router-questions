'''
Reausable function for data procesing, used by analysis notebooks as well as process code
'''

import os
import json
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
import sys
import pandas as pd
# Cargar variables de entorno desde .env
load_dotenv()

# Agregar PYTHONPATH al sys.path
sys.path.append(os.getenv('PYTHONPATH'))

RUTE_DATASETS_TEST = f"{os.getenv('PYTHONPATH')}/lab/evaluation/TestDataset/TREC-2017-LiveQA-Medical-Test-Questions-w-summaries.xml"
RUTE_NEW_DATASET_TEST = f"{os.getenv('PYTHONPATH')}/lab/evaluation/Data/"

        

def get_data_text_xml_to_json() -> None:
    '''
    Esta función transforma los datos del formato XML a un formato JSON que es más fácil de trabajar.
    '''
    # Parsear el archivo XML
    tree = ET.parse(RUTE_DATASETS_TEST)
    root = tree.getroot()

    # Listas para almacenar los datos
    data = []

    # Iterar sobre los nodos NLM-QUESTION
    for question in root.findall('.//NLM-QUESTION'):
        question_id = question.attrib.get('qid') or question.attrib.get('id')  # Obtener el ID de la pregunta
        subject = question.find('Original-Question/SUBJECT').text if question.find('Original-Question/SUBJECT') is not None else None  # Obtener el tema de la pregunta si existe
        message = question.find('Original-Question/MESSAGE').text if question.find('Original-Question/MESSAGE') is not None else None  # Obtener el mensaje de la pregunta si existe
        nist_paraphrase = question.find('NIST-PARAPHRASE').text if question.find('NIST-PARAPHRASE') is not None else None  # Obtener la paráfrasis NIST si existe
        nlm_summary = question.find('NLM-Summary').text if question.find('NLM-Summary') is not None else None  # Obtener el resumen NLM si existe
        
        if nist_paraphrase is None:
            nist_paraphrase = ','
            
        if nlm_summary is None:
            nlm_summary = ','
            
        if subject is None:
            subject = ','
        
        for focus in question.findall('.//ANNOTATIONS/FOCUS'):
            focus_text = focus.text
        
        for type_info in question.findall('.//ANNOTATIONS/TYPE'):
            type_info= type_info.text.upper()

        answer_list = []
        for ref_answer in question.findall('.//RefAnswer') or question.findall('.//ReferenceAnswer'):
            answer = {
                'aid': ref_answer.attrib.get('aid'),
                'text': ref_answer.find('ANSWER').text,
                }
            answer_list.append(answer)

        # Crear el formato de JSON
        data.append({
            'question_id': question_id,
            'subject': subject,
            'question_text': message,
            'nist_paraphrase': nist_paraphrase,
            'nlm_summary': nlm_summary,
            'focus': focus_text,
            'type_info': type_info,
            'answer_list': answer_list
        })
   ## Sustituimos algunas etiquetas
    mapeo_etiquetas = {
        ## Dentro de TREATMENT
        'LIFESTYLE_DIET': 'TREATMENT',
        'ALTERNATIVE': 'TREATMENT',
        'PREVENTION': 'TREATMENT',
        'CONTRAINDICATION': 'TREATMENT',
        'TAPERING': 'TREATMENT',
        'DOSAGE': 'TREATMENT',
        
        ## Dentro de INFOTMATION
        'STORAGE_DISPOSAL': 'INFORMATION',
        'STORAGE AND DISPOSAL': 'INFORMATION',
        'CAUSE': 'INFORMATION',
        'GENETIC CHANGES': 'INFORMATION',  
        'INHERITANCE': 'INFORMATION',
        'INDICATION': 'INFORMATION',
        'DIAGNOSIS': 'INFORMATION',
        'DIAGNOSE_ME': 'INFORMATION',
        'PROGNOSIS': 'INFORMATION',
        'SYMPTOM': 'INFORMATION',
        
        ## Dentro de OTHER
        'SIDE EFFECTS': 'OTHER',  
        'SIDE_EFFECT': 'OTHER',
        'EFFECT': 'OTHER',
        'INGREDIENT': 'OTHER',
        'PERSON_ORGANIZATION': 'OTHER',  
        'RESOURCES': 'OTHER',
        'ORGANIZATION': 'OTHER',
        'USAGE': 'OTHER',
        'COMPLICATION': 'OTHER',
        'INTERACTION': 'OTHER',
        'SUSCEPTIBILITY': 'OTHER',
        'COMPARISON': 'OTHER',
        'ASSOCIATION': 'OTHER',
        }
    for dato in data:
        tipo_info_original = dato.get('type_info')
        if tipo_info_original in mapeo_etiquetas:
            dato['type_info'] = mapeo_etiquetas[tipo_info_original]

    # Guardar como archivo JSON si no existe
    if not os.path.exists(RUTE_NEW_DATASET_TEST + 'data_test.json'):
        with open(RUTE_NEW_DATASET_TEST + 'data_test.json', 'w') as json_file:
            json.dump(data, json_file, indent=4)
        print("Archivo 'data_test.json' creado.")
    else:
        print("El archivo 'data_test.json' ya existe, no se ha creado uno nuevo.")

    # Guardar como archivo CSV si no existe
    if not os.path.exists(RUTE_NEW_DATASET_TEST + 'data_test.csv'):
        df = pd.json_normalize(data)
        df.to_csv(RUTE_NEW_DATASET_TEST + 'data_test.csv', index=False)
        print("Archivo 'data_test.csv' creado.")
    else:
        print("El archivo 'data_test.csv' ya existe, no se ha creado uno nuevo.")