import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import joblib

# Cargar datasets
data = pd.read_csv('data/combined_data.csv')

# Preprocesamiento de texto
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text

data['text'] = data['text'].apply(preprocess_text)

# División en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Vectorización de texto
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Guardar conjuntos de datos procesados y vectorizador
joblib.dump(X_train_vect, 'data/X_train_vect.pkl')
joblib.dump(X_test_vect, 'data/X_test_vect.pkl')
joblib.dump(y_train, 'data/y_train.pkl')
joblib.dump(y_test, 'data/y_test.pkl')
joblib.dump(vectorizer, 'data/vectorizer.pkl')

print("Datos de preprocesamiento y vectorizador guardados correctamente.")


