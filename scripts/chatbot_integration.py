from transformers import pipeline
import joblib

# Cargar modelo y vectorizador
model = joblib.load('../models/phishing_model.pkl')
vectorizer = joblib.load('../data/vectorizer.pkl')

# Inicializar chatbot
chatbot = pipeline('text-generation', model='gpt-2')

def check_phishing(text):
    text_vect = vectorizer.transform([text])
    prediction = model.predict(text_vect)
    return prediction[0]

def chatbot_response(user_input):
    phishing_check = check_phishing(user_input)
    if phishing_check == 1:
        return "Este mensaje parece ser un intento de phishing."
    else:
        response = chatbot(user_input, max_length=50, num_return_sequences=1)
        return response[0]['generated_text']

# Ejemplo de interacción
if __name__ == "__main__":
    while True:
        user_input = input("Usuario: ")
        if user_input.lower() == 'salir':
            break
        response = chatbot_response(user_input)
        print(f'Chatbot: {response}')
