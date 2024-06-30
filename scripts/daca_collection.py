import requests
import pandas as pd

# URLs de datasets de phishing y correos legítimos
phishing_urls = [
    'https://example.com/phishing_dataset1.csv',
    'https://example.com/phishing_dataset2.csv'
]

legit_urls = [
    'https://example.com/legit_dataset1.csv',
    'https://example.com/legit_dataset2.csv'
]

# Función para descargar y combinar datasets
def download_datasets(urls):
    data_frames = []
    for url in urls:
        response = requests.get(url)
        df = pd.read_csv(url)
        data_frames.append(df)
    combined_df = pd.concat(data_frames, ignore_index=True)
    return combined_df

phishing_data = download_datasets(phishing_urls)
legit_data = download_datasets(legit_urls)

# Guardar datasets combinados
phishing_data.to_csv('../data/phishing_data.csv', index=False)
legit_data.to_csv('../data/legit_data.csv', index=False)
