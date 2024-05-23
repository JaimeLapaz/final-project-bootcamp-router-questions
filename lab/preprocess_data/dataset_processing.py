'''
Reausable function for data procesing, used by analysis notebooks as well as process code
'''

import os
import json
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
import sys
import pandas as pd
import re
# Cargar variables de entorno desde .env
load_dotenv()

# Agregar PYTHONPATH al sys.path
sys.path.append(os.getenv('PYTHONPATH'))

RUTE_DATASETS_1 = f"{os.getenv('PYTHONPATH')}/lab/preprocess_data/TrainingDatasets/"
RUTE_DATASETS_2 = f"{os.getenv('PYTHONPATH')}/lab/preprocess_data/TrainingDatasets/"
RUTE_NEW_DATASET = f"{os.getenv('PYTHONPATH')}/lab/preprocess_data/Data/"

##Función que combina los datasets de train
def combine_datasets_train(datasets_1: str, datasets_2: str, new_datasets: str) -> None:
    '''
    Esta función combina los dos archivos xml de datos que tenemos para ponerlos en un solo archivo,
    de esta forma conseguimos pasarlo todo luego a json directamente.
    '''
    # Parsea los documentos XML
    tree1 = ET.parse(RUTE_DATASETS_1 + datasets_1)
    tree2 = ET.parse(RUTE_DATASETS_2 + datasets_2)

    root1 = tree1.getroot()
    root2 = tree2.getroot()

    # Supongamos que queremos agregar todos los elementos de root2 a root1
    for child in root2:
        root1.append(child)

     # Verifica si el archivo nuevo ya existe
    if not os.path.exists(RUTE_NEW_DATASET + new_datasets):
        # Escribe el nuevo documento XML combinado solo si no existe
        tree1.write(RUTE_NEW_DATASET + new_datasets)
        print(f"Archivo '{new_datasets}' creado.")
    else:
        print(f"El archivo '{new_datasets}' ya existe, no se ha creado uno nuevo.")


## Limpieza básica del texto antes de guardarlo
def text_clean_data_train(text: str) -> str:
    '''
    Con esta función limpiamos un poco los datos antes de pasarlos a guardar, quitaremos algunas configuraciones raras que aparecen.
    '''
    # Reemplazar múltiples espacios consecutivos por un solo espacio
    clean_text = re.sub(r'\s+', ' ', text)
    # Eliminar espacios adicionales al inicio y al final del texto
    clean_text = clean_text.strip()
    return clean_text


##Vamos a pasar los datos a un archivo JSON y csv para trabajar con el que mejor nos venga
def get_data_train_xml_to_json(datasets: str)->None:
    '''
    Esta función transforma los datos del formato XML a un formato JSON que es más fácil de trabajar.
    '''
    # Parsear el archivo XML
    tree = ET.parse(RUTE_NEW_DATASET + datasets)
    root = tree.getroot()

    # Listas para almacenar los datos
    data = []

    # Iterar sobre los nodos NLM-QUESTION
    for question in root.findall('.//NLM-QUESTION'):
        question_id = question.attrib.get('questionid') or question.attrib.get('qid') # Como hay 2 id buscamos el que exista
        subject = question.find('SUBJECT').text if question.find('SUBJECT') is not None else None  # Añadir 'subject' si existe
        message = question.find('MESSAGE').text
        
       # Iterar sobre los nodos NLM-QUESTION
    for question in root.findall('.//NLM-QUESTION'):
        question_id = question.attrib.get('questionid') or question.attrib.get('qid')
        subject = question.find('SUBJECT').text if question.find('SUBJECT') is not None else None  # Añadir 'subject' si existe
        message = question.find('MESSAGE').text
        if subject is None:
            subject = ','
        if message is None:
            message = subject
        
        question_text = text_clean_data_train(message)
        # Iterar sobre los nodos SUB-QUESTION dentro de cada NLM-QUESTION
        for subquestion in question.findall('.//SUB-QUESTION'):
        #subquestion_id = subquestion.attrib.get('subqid')
            focus = subquestion.find('.//FOCUS').text
            type_info = subquestion.find('.//TYPE').text.upper()
        
            
            # Iterar sobre los nodos ANSWER dentro de cada SUB-QUESTION
            for answer in subquestion.findall('.//ANSWER'):
                #answer_id = answer.attrib.get('answerid')
                answer_text = answer.text
                answer_text_clean = text_clean_data_train(answer_text)
        
        # Creamos el formato de JSON que queremos el documento
        data.append({'question_id': question_id,
                            'question_text': question_text,
                            'subject': subject,
                            'focus': focus,
                            'type_info': type_info,                            
                            'answer_text': answer_text_clean})
    
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
         
    # Verifica si el archivo JSON ya existe
    if not os.path.exists(RUTE_NEW_DATASET + 'data_train.json'):
        # Crear archivo JSON solo si no existe
        with open(RUTE_NEW_DATASET + 'data_train.json', 'w') as json_file:
            json.dump(data, json_file, indent=4)
        print("Archivo 'data.json' creado.")
    else:
        print("El archivo 'data.json' ya existe, no se ha creado uno nuevo.")
    
    if not os.path.exists(RUTE_NEW_DATASET + 'data_train.csv'):
        # Crear archivo csv
        df = pd.DataFrame(data)
        df.to_csv(RUTE_NEW_DATASET + 'data_train.csv', index=False)
        print("Archivo 'data_train.csv' creado.")
    else:
        print("El archivo 'data_train.csv' ya existe, no se ha creado uno nuevo.")
        