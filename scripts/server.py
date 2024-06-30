from flask import Flask, request, jsonify
import joblib
from transformers import pipeline

app = Flask(__name__)

# Cargar modelo y vectorizador
model = joblib.load('../models/phishing_model.pkl')
vectorizer = joblib.load('../data/vectorizer.pkl')
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
