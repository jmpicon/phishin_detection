import pandas as pd

# Ruta del archivo CSV original
data_path = 'data/phishing_data.csv'

# Ruta del archivo CSV limpio
clean_data_path = 'data/phishing_data_clean.csv'

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

# Convertir la columna 'url' a cadenas de texto y eliminar valores NaN
df['url'] = df['url'].astype(str)

# Eliminar filas con valores vacíos en 'url' o 'label'
df = df[df['url'].str.strip() != '']
df = df.dropna(subset=['url', 'label'])

# Guardar el DataFrame limpio
df.to_csv(clean_data_path, index=False)
print(f"Archivo CSV limpio guardado en {clean_data_path}")
