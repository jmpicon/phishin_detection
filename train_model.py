import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def evaluate_model():
    # Cargar datos desde el archivo limpio
    data_path = 'data/phishing_data_clean.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"El archivo {data_path} no se encontró.")
    
    df = pd.read_csv(data_path)
    
    if 'url' not in df.columns or 'label' not in df.columns:
        raise KeyError("Las columnas 'url' y 'label' deben estar presentes en el archivo CSV.")

    df['url'] = df['url'].astype(str)
    df = df.dropna(subset=['url', 'label'])

    X = df['url']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = make_pipeline(TfidfVectorizer(), RandomForestClassifier(n_estimators=100, random_state=42))

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("Reporte de clasificación:\n", classification_report(y_test, y_pred))
    print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))
    
    auc = roc_auc_score(y_test, y_proba)
    print(f"AUC: {auc:.2f}")
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

    cross_val_scores = cross_val_score(pipeline, X, y, cv=5)
    print(f"Cross-Validation Scores: {cross_val_scores}")
    print(f"Mean Cross-Validation Score: {cross_val_scores.mean()}")

if __name__ == "__main__":
    evaluate_model()


