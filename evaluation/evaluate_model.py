import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Cargar conjuntos de datos procesados y modelo
X_test_vect = joblib.load('../data/X_test_vect.pkl')
y_test = joblib.load('../data/y_test.pkl')
model = joblib.load('../models/phishing_model.pkl')

# Generar curvas ROC
y_pred_proba = model.predict_proba(X_test_vect)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plotear curva ROC
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.savefig('../evaluation/roc_curve.png')
plt.show()
