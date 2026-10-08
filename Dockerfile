FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

WORKDIR /app

# 1. Copiamos ESPECÍFICAMENTE el TOML de la API
COPY backend/pyproject.toml ./
RUN uv pip install --system -r pyproject.toml

# 2. Copiamos ESPECÍFICAMENTE el main.py de la API
COPY backend/main.py ./

# 3. Copiamos la carpeta Datos de la raíz
COPY Datos/ ./Datos/

# 4. Levantamos la API
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}