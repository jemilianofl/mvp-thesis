# Usar una imagen oficial de Python ligera y compatible con tu proyecto
FROM python:3.12-slim

# Instalar dependencias del sistema recomendadas para GeoPandas y librerías espaciales
RUN apt-get update && apt-get install -y \
    build-essential \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Instalar el gestor de paquetes ultra rápido 'uv'
RUN pip install uv

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar el archivo de dependencias del backend
COPY pyproject.toml ./

# Usar uv para instalar las dependencias directamente en el sistema del contenedor
RUN uv pip install --system -r pyproject.toml

# Copiar el script principal de tu API
COPY main.py ./

# Comando para ejecutar FastAPI usando el puerto dinámico que asigna Render
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}