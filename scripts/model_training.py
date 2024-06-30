import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Cargar conjuntos de datos procesados
X_train_vect = joblib.load('data/X_train_vect.pkl')
y_train = joblib.load('data/y_train.pkl')
X_test_vect = joblib.load('data/X_test_vect.pkl')
y_test = joblib.load('data/y_test.pkl')

# Entrenamiento del modelo
model = LogisticRegression()
model.fit(X_train_vect, y_train)

# Evaluación del modelo
y_pred = model.predict(X_test_vect)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Guardar modelo
joblib.dump(model, 'models/phishing_model.pkl')

print("Modelo guardado correctamente.")

