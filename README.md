# Password Security Analysis

An academic cybersecurity and machine learning project investigating whether learned password-pattern features can identify predictable passwords that conventional rule-based checks may miss.

## Approach
- Dataset preparation and validation.
- Password feature extraction.
- Rule-based baseline and machine learning comparison.
- Evaluation, robustness checks and an interactive Streamlit interface.

## Quick start
The application lives in [password-security-analysis/](password-security-analysis/).

```sh
cd password-security-analysis
docker compose up app
```

Open http://localhost:8501. See the [technical README](password-security-analysis/README.md) for setup and the experimental workflow.

Use synthetic examples rather than real credentials. This is a research prototype, not a guarantee of password security.
