FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

WORKDIR /app

# 1. Copiamos los archivos que están sueltos en la raíz
COPY pyproject.toml ./
RUN uv pip install --system -r pyproject.toml

COPY main.py ./

# 2. Copiamos la carpeta Datos para que la API tenga información
COPY Datos/ ./Datos/

# 3. Arrancamos el servidor
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}