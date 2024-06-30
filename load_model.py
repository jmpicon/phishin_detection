import joblib
import os

# Ruta del modelo
model_path = '../models/phishing_model.pkl'

# Verificar si el archivo del modelo existe
if not os.path.exists(model_path):
    raise FileNotFoundError(f"El archivo del modelo no se encontró: {model_path}")

# Cargar el modelo
pipeline = joblib.load(model_path)

# Datos de ejemplo para predecir
new_data = [
    "This is your last chance to update your account information.",
    "Team meeting tomorrow at 10 AM."
]

# Realizar predicciones
predictions = pipeline.predict(new_data)
print("Predicciones:", predictions)

