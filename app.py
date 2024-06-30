from flask import Flask, render_template, request, redirect, url_for
from flask_socketio import SocketIO, emit
import joblib
import os
import pandas as pd

app = Flask(__name__)
socketio = SocketIO(app)

# Ruta del modelo
model_path = 'models/phishing_model.pkl'

# Verificar si el archivo del modelo existe antes de cargarlo
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = None
    print(f"El archivo del modelo no se encontró en {model_path}.")

user_context = {}

@app.route('/')
def home():
    return render_template('index.html')

@socketio.on('message')
def handle_message(message):
    user_id = request.sid
    if user_id not in user_context:
        user_context[user_id] = {"stage": 0, "url": ""}
    
    if user_context[user_id]["stage"] == 0:
        user_context[user_id]["url"] = message
        user_context[user_id]["stage"] = 1
        emit('response', "¿Puedes confirmar si esta URL es sospechosa? (sí/no)")
    elif user_context[user_id]["stage"] == 1:
        if message.lower() in ['sí', 'si', 'yes']:
            user_context[user_id]["stage"] = 2
            emit('response', "Gracias por confirmar. Estamos procesando la URL...")
            url = user_context[user_id]["url"]
            if model:
                prediction = model.predict([url])[0]
                response = 'Phishing' if prediction == 1 else 'Not Phishing'
                emit('response', f"La URL es detectada como: {response}")
                
                # Guardar el nuevo dato
                new_data = {'url': [url], 'label': [1 if response == 'Phishing' else 0]}
                df = pd.DataFrame(new_data)
                data_path = 'data/phishing_data_clean.csv'
                if os.path.exists(data_path):
                    df.to_csv(data_path, mode='a', header=False, index=False)
                else:
                    df.to_csv(data_path, index=False)
                
                user_context[user_id]["stage"] = 0
            else:
                emit('response', 'Modelo no cargado correctamente.')
                user_context[user_id]["stage"] = 0
        else:
            emit('response', "Por favor, ingresa una URL para verificar.")
            user_context[user_id]["stage"] = 0

@app.route('/retrain')
def retrain():
    train_model()
    global model
    model = joblib.load(model_path)
    return redirect(url_for('home'))

def train_model():
    # Cargar datos desde el archivo limpio
    data_path = 'data/phishing_data_clean.csv'
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print("Columnas del DataFrame:", df.columns)  # Imprimir las columnas del DataFrame
    else:
        raise FileNotFoundError(f"El archivo {data_path} no se encontró.")
    
    # Asumimos que las columnas son 'url' y 'label'
    if 'url' not in df.columns or 'label' not in df.columns:
        raise KeyError("Las columnas 'url' y 'label' deben estar presentes en el archivo CSV.")

    # Convertir la columna 'url' a cadenas de texto y eliminar valores NaN
    df['url'] = df['url'].astype(str)
    df = df.dropna(subset=['url', 'label'])

    X = df['url']
    y = df['label']

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_url, X_url, y_label, y_label = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear un pipeline que incluye TfidfVectorizer y RandomForestClassifier
    pipeline = make_pipeline(TfidfVectorizer(), RandomForestClassifier(n_estimators=100, random_state=42))

    # Entrenar el modelo
    pipeline.fit(X_train, y_train)

    # Evaluar el modelo
    y_pred = pipeline.predict(X_test)
    print("Reporte de clasificación:\n", classification_report(y_test, y_pred))
    print("Precisión del modelo:", accuracy_score(y_test, y_pred))
    print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))

    # Crear el directorio si no existe
    model_dir = 'models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Guardar el modelo
    model_path = os.path.join(model_dir, 'phishing_model.pkl')
    joblib.dump(pipeline, model_path)

    print(f"Modelo guardado exitosamente en {model_path}.")

if __name__ == "__main__":
    socketio.run(app, debug=True)


