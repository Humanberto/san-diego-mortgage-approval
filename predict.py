"""Web service: send an applicant as JSON, get back the approval probability.

Run from the project folder:
    uv run uvicorn predict:app --host 0.0.0.0 --port 9696
Then open http://localhost:9696 for the form, or /docs for the raw API.
"""
import os
import pickle
from typing import Any

import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse

MODEL_PATH = os.getenv('MODEL_PATH', 'model_2025.bin')
with open(MODEL_PATH, 'rb') as f_in:
    bundle = pickle.load(f_in)

app = FastAPI(title='San Diego mortgage approval')


def prepare(applicant: dict[str, Any]) -> dict[str, Any]:
    """Apply the same steps as train.py to one applicant."""
    record = {}
    for col in bundle['categorical']:
        value = applicant.get(col)
        record[col] = 'missing' if value is None else str(value)
    for col in bundle['numerical']:
        value = applicant.get(col)
        value = bundle['medians'][col] if value is None else float(value)
        if col in bundle['log_columns']:
            value = float(np.log1p(max(value, 0)))
        record[col] = value
    return record


@app.get('/', include_in_schema=False)
def form() -> FileResponse:
    """The simple web form (index.html) that calls /predict."""
    return FileResponse('index.html')


@app.get('/health')
def health() -> dict[str, Any]:
    return {'status': 'ok', 'model_year': bundle['year'], 'test_auc': bundle['test_auc']}


@app.post('/predict')
def predict(applicant: dict[str, Any]) -> dict[str, Any]:
    X = bundle['dv'].transform([prepare(applicant)])
    probability = float(bundle['model'].predict_proba(X)[0, 1])
    return {
        'approval_probability': round(probability, 3),
        'approved': probability >= 0.5,
    }
