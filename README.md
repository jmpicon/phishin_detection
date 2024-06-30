# Detección Proactiva de Phishing mediante IA e Integración de un Chatbot

## Descripción

Este proyecto tiene como objetivo desarrollar un modelo avanzado basado en aprendizaje automático para la detección y prevención de ataques de phishing, y la integración de un chatbot basado en modelos de lenguaje de última generación (LLMs) para asistencia y soporte en tiempo real.

## Estructura del Proyecto

phishing-detection/
│
├── data/
│ ├── phishing_data.csv
│ └── legit_data.csv
│
├── scripts/
│ ├── data_collection.py
│ ├── preprocessing.py
│ ├── model_training.py
│ ├── chatbot_integration.py
│ └── server.py
│
├── evaluation/
│ ├── evaluate_model.py
│ └── roc_curve.png
│
├── notebook/
│ └── analysis_notebook.ipynb
│
├── reports/
│ └── final_report.md
│
├── requirements.txt
├── README.md
└── .gitignore




## Instalación

1. Clona este repositorio.
2. Instala las dependencias con `pip install -r requirements.txt`.
3. Ejecuta los scripts en el orden especificado para recolectar, preprocesar datos, entrenar el modelo e implementar el servidor y chatbot.

## Uso

1. Ejecuta `python scripts/server.py` para iniciar el servidor.
2. Interactúa con el chatbot a través de la API REST.

## Evaluación

Ejecuta `python evaluation/evaluate_model.py` para evaluar el modelo y generar la curva ROC.

## Autores

José Manuel Picón Giménez
