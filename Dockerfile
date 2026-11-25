# APEX Industrial AI Reliability Engine - single-container image for hosting.
# Stage 1 builds the React dashboard; stage 2 runs the FastAPI backend, which
# also serves the built dashboard, so one URL gives the whole site.

FROM node:20-slim AS dashboard
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt
COPY backend/ backend/
COPY apex_core/ apex_core/
COPY data/ data/
COPY --from=dashboard /app/frontend/dist frontend/dist

# Render supplies $PORT; Hugging Face Spaces and local runs default to 7860.
EXPOSE 7860
CMD ["sh", "-c", "uvicorn backend.server:app --host 0.0.0.0 --port ${PORT:-7860}"]
