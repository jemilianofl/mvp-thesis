FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

# Establecer directorio base
WORKDIR /app

# 1. Copiar e instalar dependencias indicando la carpeta backend
COPY backend/pyproject.toml ./backend/
RUN cd backend && uv pip install --system -r pyproject.toml

# 2. Copiar el script principal a su carpeta
COPY backend/main.py ./backend/

# 3. Copiar la carpeta Datos para que la API tenga información que servir
COPY Datos/ ./Datos/

# 4. Moverse a la carpeta backend para arrancar el servidor
WORKDIR /app/backend

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}