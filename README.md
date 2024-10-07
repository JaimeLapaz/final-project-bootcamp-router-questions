# Disease Question Router / Enrutador de Preguntas sobre Enfermedades 

Este proyecto está disponible en:
- [English](#english-version)
- [Español](#versión-en-español)

---

## English Version

### Table of Contents

1. [Project Description](#project-description)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Steps to Follow](#steps-to-follow)
5. [Usage](#usage)
6. [Results](#results)
7. [Future Improvements](#future-improvements)
8. [Conclusion](#conclusion)

### Project Description

The **Disease Question Router** classifies medical questions and provides an answer if the response is available in its database. For this, the `bert-base-uncased` model from Huggingface was fine-tuned using domain-specific medical data.

The project is divided into two main components:
- **Model training:** Fine-tuning a pre-trained model to improve the classification of medical questions.
- **Interactive interface:** An interface where users can interact with the model in real-time, asking questions and receiving answers.

### Requirements

- Python 3.7+
- [Huggingface Transformers](https://huggingface.co/transformers)
- [PyTorch](https://pytorch.org/)
- [scikit-learn](https://scikit-learn.org/)
- [MLflow](https://mlflow.org/)
- Additional dependencies are listed in `requirements.txt`

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/JaimeLapaz/final-project-bootcamp-router-questions.git
   cd DiseaseQuestionRouter
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the pre-trained Huggingface model:
   ```bash
   from transformers import BertTokenizer, BertForSequenceClassification
   model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
   ```

4. Set the necessary environment variables for model training and storage.

### Steps to Follow

1. Create the `.env` file and set the project path:
   ```bash
   PYTHONPATH=Your/project/path
   ```
2. Launch the MLflow server:
   ```bash
   mlflow server --host 127.0.0.1 --port 5050
   ```
3. Train the model:
   ```bash
   python lab/main.py
   ```
4. Launch the app after the model is trained:
   ```bash
   python my_project/app.py
   ```

### Usage

#### Model Training
You can train the model by running the following script:

```bash
python lab/main.py
```

This script:
- Preprocesses the data.
- Splits the dataset into training and validation sets.
- Fine-tunes the `BERT` pre-trained model using the provided medical data.
- Saves the fine-tuned model and performance metrics.

#### Interactive Interface

Once the model is trained, you can launch the interactive interface to ask medical questions:

```bash
python my_project/app.py
```

### Results

During the training phase, the following performance results were obtained:

- **Accuracy:** 0.80
- **F1 Score:** 0.66
- **Recall:** 0.65
- **Precision:** 0.71
- **Loss:** 0.03

In the evaluation phase, the model achieved a 49.04% accuracy rate, with an average response time of 0.20 seconds.

### Future Improvements

There are several areas for improvement in this project:
1. **Expanding the database** to include more questions and answers, thereby improving the model's generalization capabilities.
2. **Text generation models** to automatically generate answers,

 instead of relying solely on predefined responses.
3. **Human validation** to continue training the model with new question classifications and improve its accuracy over time.

### Conclusion

This project demonstrates the practical application of Natural Language Processing and Artificial Intelligence techniques in the healthcare field. Although functional, future improvements will allow for greater precision and utility in the medical domain.

---

## Versión en Español

### Tabla de Contenidos

1. [Descripción del Proyecto](#descripción-del-proyecto)
2. [Requisitos](#requisitos)
3. [Instalación](#instalación)
4. [Pasos a seguir](#pasos-a-seguir)
5. [Uso](#uso)
6. [Resultados](#resultados)
7. [Mejoras Futuras](#mejoras-futuras)
8. [Conclusión](#conclusión)

### Descripción del Proyecto

El **Enrutador de Preguntas sobre Enfermedades** clasifica preguntas médicas y, si la respuesta está disponible en su base de datos, proporciona una respuesta automáticamente. Para ello, se utiliza el modelo `bert-base-uncased` de Huggingface, ajustado con datos específicos del dominio de la salud. 

El proyecto está dividido en dos partes principales:
- **Entrenamiento del modelo:** Finetuning de un modelo preentrenado para mejorar la clasificación de preguntas médicas.
- **Interfaz interactiva:** Implementación de una interfaz para que los usuarios puedan interactuar con el modelo en tiempo real, realizando preguntas y obteniendo respuestas.

### Requisitos

- Python 3.7+
- [Huggingface Transformers](https://huggingface.co/transformers)
- [PyTorch](https://pytorch.org/)
- [scikit-learn](https://scikit-learn.org/)
- [MLflow](https://mlflow.org/)
- Otras dependencias necesarias están en el archivo `requirements.txt`

### Instalación

1. Clona este repositorio:
   ```bash
   git clone https://github.com/JaimeLapaz/final-project-bootcamp-router-questions.git
   cd EnrutadorEnfermedades
   ```

2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

3. Descarga el modelo preentrenado de Huggingface:
   ```bash
   from transformers import BertTokenizer, BertForSequenceClassification
   model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
   ```

4. Configura las variables de entorno necesarias para el entrenamiento y almacenamiento de los modelos.

### Pasos a seguir

1. Crea el archivo `.env` y coloca la ruta del proyecto:
   ```bash
   PYTHONPATH=Tu/ruta/proyecto
   ```
2. Lanza el servidor MLflow:
   ```bash
   mlflow server --host 127.0.0.1 --port 5050
   ```
3. Entrena el modelo:
   ```bash
   python lab/main.py
   ```
4. Lanza la aplicación una vez entrenado el modelo:
   ```bash
   python my_project/app.py
   ```

### Uso

#### Entrenamiento del Modelo
El entrenamiento del modelo se puede realizar ejecutando el siguiente script:

```bash
python lab/train.py
```

Este script:
- Preprocesa los datos.
- Divide el conjunto de datos en entrenamiento y validación.
- Ajusta el modelo `BERT` preentrenado con los datos médicos proporcionados.
- Guarda el modelo ajustado y las métricas de rendimiento.

#### Interfaz Interactiva

Una vez entrenado el modelo, se puede iniciar la interfaz interactiva para hacer preguntas médicas:

```bash
python my_project/app.py
```

### Resultados

Durante el entrenamiento, se lograron los siguientes resultados en términos de rendimiento:

- **Exactitud (Accuracy):** 0.80
- **F1 Score:** 0.66
- **Recall:** 0.65
- **Precisión:** 0.71
- **Pérdida (Loss):** 0.03

En la fase de evaluación, el modelo alcanzó un 49.04% de aciertos con un tiempo de respuesta promedio de 0.20 segundos.

### Mejoras Futuras

El proyecto tiene varias áreas de mejora, como:
1. **Ampliación de la base de datos** para incluir más preguntas y respuestas, mejorando así la capacidad de generalización del modelo.
2. **Generación de respuestas automáticas** utilizando modelos generativos en lugar de depender únicamente de respuestas predefinidas.
3. **Validación humana** para seguir entrenando el modelo con nuevas clasificaciones.

### Conclusión

Este proyecto demuestra el uso práctico de las técnicas de procesamiento del lenguaje natural y la inteligencia artificial en el campo de la salud. Aunque ya es funcional, las mejoras futuras permitirán una mayor precisión y utilidad en el ámbito médico.

---
