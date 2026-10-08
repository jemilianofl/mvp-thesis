FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

WORKDIR /app

# Copiamos primero el TOML desde la carpeta backend de tu repo
COPY backend/pyproject.toml ./
RUN uv pip install --system -r pyproject.toml

# Copiamos el main.py desde la carpeta backend de tu repo
COPY backend/main.py ./

# Copiamos la carpeta Datos de tu repo
COPY Datos/ ./Datos/

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}