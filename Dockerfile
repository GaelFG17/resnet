FROM python:3.11-slim

WORKDIR /app

# Copiar los archivos necesarios
COPY requirements.txt ./

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY . .

# Exponer el puerto en el que correrá la app
EXPOSE 5000

# Comando para ejecutar la aplicación
CMD ["python", "app.py"]
