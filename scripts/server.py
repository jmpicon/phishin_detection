from flask import Flask, request, jsonify
import joblib
from transformers import pipeline

app = Flask(__name__)

# Función para cargar un archivo con verificación
def load_file(file_path):
    try:
        return joblib.load(file_path)
    except EOFError:
        print(f"Error: El archivo {file_path} está corrupto o incompleto.")
        return None

# Cargar modelo y vectorizador
model = load_file('models/phishing_model.pkl')
vectorizer = load_file('data/vectorizer.pkl')

# Verificar si los archivos se cargaron correctamente
if model is None or vectorizer is None:
    print("Error al cargar el modelo o el vectorizador.")
    exit()

chatbot = pipeline('text-generation', model='gpt-2')

def check_phishing(text):
    text_vect = vectorizer.transform([text])
    prediction = model.predict(text_vect)
    return prediction[0]

@app.route('/chatbot', methods=['POST'])
def chatbot_response():
    user_input = request.json['message']
    phishing_check = check_phishing(user_input)
    if phishing_check == 1:
        return jsonify({"response": "Este mensaje parece ser un intento de phishing."})
    else:
        response = chatbot(user_input, max_length=50, num_return_sequences=1)
        return jsonify({"response": response[0]['generated_text']})

if __name__ == '__main__':
    app.run(debug=True)



