import os
import sys
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
from dotenv import load_dotenv
import time
import random
import numpy as np
import joblib

load_dotenv()


# Agregar PYTHONPATH al sys.path
sys.path.append(os.getenv('PYTHONPATH'))

RUTE_MODEL = f"{os.getenv('PYTHONPATH')}"
MODEL_NAME = "bert-base-uncased"
BATCH_SIZE_MODEL = 16
EPOCH_SIZE_MODEL = 5
NAME_MODEL_SAVE = f"{MODEL_NAME}_{BATCH_SIZE_MODEL}_{EPOCH_SIZE_MODEL}"
PATH_MY_MODEL = f"{os.getenv('PYTHONPATH')}/lab/my_model/{NAME_MODEL_SAVE}"

class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Configurar semillas para reproducibilidad
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_confusion_matrix(y_true, y_pred, labels, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion Matrix")
    plt.savefig(filename)
    plt.close()
    print(f"La matriz de confusión de ha guardado como {filename}")
    

def train_model(train_questions, val_questions, train_labels, val_labels, label_encoder):
    # Declaramos la semilla
    set_seed(42)
    
    # Parámetros a optimizar
    batch_size_model = BATCH_SIZE_MODEL
    epoch_size = EPOCH_SIZE_MODEL
    name_my_model = NAME_MODEL_SAVE
    
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    train_encodings = tokenizer(train_questions, truncation=True, padding=True)
    val_encodings = tokenizer(val_questions, truncation=True, padding=True)
    train_dataset = CustomDataset(train_encodings, train_labels)
    val_dataset = CustomDataset(val_encodings, val_labels)
    
    
    # Definir DataLoader para conjunto de entrenamiento
    train_loader = DataLoader(train_dataset, batch_size=batch_size_model, shuffle=True)
    
    # Fine-tuning del modelo BERT para clasificación de secuencia
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(label_encoder.classes_))
    
    # Entrenamiento del modelo con MLflow
    with mlflow.start_run() as run:
        mlflow.log_param("model_name", MODEL_NAME)
        start_time = time.time()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)
        model.train()
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        epochs = epoch_size

        for epoch in range(epochs):
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
                # Registrar métrica de pérdida
                mlflow.log_metric("loss", loss.item(), step=epoch)

        # Evaluación del modelo
        model.eval()
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

        all_val_labels = []
        all_predictions = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_val_labels, all_predictions)
        f1 = f1_score(all_val_labels, all_predictions, average='macro')
        precision = precision_score(all_val_labels, all_predictions, average='macro')
        recall = recall_score(all_val_labels, all_predictions, average='macro')
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", round(f1, 2))
        mlflow.log_metric("precision", round(precision, 2))
        mlflow.log_metric("recall", round(recall, 2))

        # Guardar modelo como artefacto de MLflow
        mlflow.pytorch.log_model(model, name_my_model)

        # Guardar el modelo localmente
        model_save_path = f"{RUTE_MODEL}/lab/my_model/{name_my_model}"
        model_save_path2 = f"{RUTE_MODEL}/my_project/model/{name_my_model}"
        model.save_pretrained(model_save_path)
        model.save_pretrained(model_save_path2)

        # Guardar el laber_encoder
        joblib.dump(label_encoder, f"{RUTE_MODEL}/lab/my_model/label_encoder.pkl")
        joblib.dump(label_encoder, f"{RUTE_MODEL}/my_project/model/label_encoder.pkl")
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        mlflow.log_metric("training_time", elapsed_time)
        mlflow.log_metric("train_size", len(train_dataset))
        mlflow.log_metric("val_size", len(val_dataset))

        # Imprimir métricas y matriz de confusión
        print("Accuracy:", accuracy)
        print("F1-score:", round(f1, 2))
        print("Recall:", round(recall, 2))
        print("Precision:", round(precision, 2))
        save_confusion_matrix(all_val_labels, all_predictions, label_encoder.classes_, "confusion_matrix.png")