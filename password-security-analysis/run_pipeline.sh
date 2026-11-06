#!/bin/sh
# Runs section 2 of the README (steps 1-5b) on the frozen samples in data/samples/.
# Used by Docker; you can also run it yourself: sh run_pipeline.sh
set -e
python -m src.data.clean_dataset
python -m src.data.validate_dataset
python -m src.features.password_features
python -m src.models.rule_based
python -m src.models.train_ml --ablation
