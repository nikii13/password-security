# Same Python + same library versions for everyone who runs the project.
FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the code. Data is NOT baked into the image: it is mounted as a
# volume at runtime (see docker-compose.yml), so rockyou.txt never ends up
# inside an image that could be pushed somewhere.
COPY src/ src/
COPY tests/ tests/

# Default: run the whole data pipeline
CMD ["sh", "-c", "python -m src.data.sample_dataset && python -m src.data.clean_dataset && python -m src.data.validate_dataset && python -m src.features.password_features"]
