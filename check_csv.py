import pandas as pd

# Ruta del archivo CSV
data_path = 'data/phishing_data.csv'

# Intentar cargar el archivo CSV
try:
    df = pd.read_csv(data_path)
except pd.errors.ParserError as e:
    print(f"Error al cargar el archivo CSV: {e}")
    df = pd.read_csv(data_path, on_bad_lines='skip')  # Ignorar líneas problemáticas

# Mostrar las primeras filas y las columnas del DataFrame
print("Primeras filas del DataFrame:")
print(df.head())
print("Columnas del DataFrame:", df.columns)

# Eliminar filas con valores NaN en 'url' o 'label'
df = df.dropna(subset=['url', 'label'])

# Convertir la columna 'url' a cadenas de texto
df['url'] = df['url'].astype(str)

# Guardar el DataFrame limpio
df.to_csv('data/phishing_data_clean.csv', index=False)
