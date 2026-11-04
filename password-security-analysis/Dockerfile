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

# Default: run the pipeline on the FROZEN samples in data/samples/
# (section 2 of the README). Sampling (section 1) is not re-run on purpose.
CMD ["sh", "-c", "python -m src.data.clean_dataset && python -m src.data.validate_dataset && python -m src.features.password_features && python -m src.models.rule_based"]
