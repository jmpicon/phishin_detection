import joblib

# Cargar el vectorizador
vectorizer = joblib.load('data/vectorizer.pkl')

# Verificar atributos del vectorizador
print("Características del vectorizador:")
print(f"Número de características: {len(vectorizer.get_feature_names_out())}")
print(f"Primeras 10 características: {vectorizer.get_feature_names_out()[:10]}")
