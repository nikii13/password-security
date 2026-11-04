"""Create a reproducible sample from a CSV.

For very large files, this uses chunked reading and reservoir sampling, so it
does not load the entire dataset into memory. Adapt INPUT_PATH and the column
names to the source dataset. Sampling is uniform over rows; if labels/classes
are imbalanced, use a stratified sampling strategy instead.
"""
from pathlib import Path
import random
import pandas as pd

INPUT_PATH = Path("data/raw/dataset_original.csv")
OUTPUT_PATH = Path("data/samples/password_sample.csv")
CHUNK_SIZE = 100_000
SAMPLE_SIZE = 100_000
SEED = 42

def reservoir_sample_csv(input_path: Path, sample_size: int, chunksize: int, seed: int):
    rng = random.Random(seed)
    reservoir = []
    seen = 0

    for chunk in pd.read_csv(input_path, chunksize=chunksize):
        for row in chunk.to_dict(orient="records"):
            seen += 1
            if len(reservoir) < sample_size:
                reservoir.append(row)
            else:
                j = rng.randrange(seen)
                if j < sample_size:
                    reservoir[j] = row

    return pd.DataFrame(reservoir), seen

def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"No se encuentra {INPUT_PATH}. Coloca ahí el CSV original."
        )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample, total_rows = reservoir_sample_csv(
        INPUT_PATH, SAMPLE_SIZE, CHUNK_SIZE, SEED
    )
    sample.to_csv(OUTPUT_PATH, index=False)
    print(f"Filas leídas: {total_rows:,}")
    print(f"Muestra guardada: {len(sample):,} -> {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
