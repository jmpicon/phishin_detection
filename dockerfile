# Usar una imagen base de Python
FROM python:3.12

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar los archivos de requisitos
COPY requirements.txt .

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto de la aplicación
COPY . .

# Exponer el puerto de la aplicación Flask
EXPOSE 5000

# Comando para ejecutar la aplicación
CMD ["python", "app.py"]
