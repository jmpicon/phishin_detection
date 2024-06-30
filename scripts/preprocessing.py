import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Cargar datasets
phishing_data = pd.read_csv('../data/phishing_data.csv')
legit_data = pd.read_csv('../data/legit_data.csv')

# Añadir columna de etiquetas
phishing_data['label'] = 1
legit_data['label'] = 0

# Combinar datasets
data = pd.concat([phishing_data, legit_data], ignore_index=True)

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

# Guardar conjuntos de datos procesados
joblib.dump(X_train_vect, '../data/X_train_vect.pkl')
joblib.dump(X_test_vect, '../data/X_test_vect.pkl')
joblib.dump(y_train, '../data/y_train.pkl')
joblib.dump(y_test, '../data/y_test.pkl')
joblib.dump(vectorizer, '../data/vectorizer.pkl')
