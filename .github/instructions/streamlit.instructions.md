---
description: "Use when working on the frontend Streamlit application in the app/ folder."
applyTo:
  - "app/**/*.py"
---

## Streamlit Development Conventions

- **Inference Integration**: The frontend acts as a wrapper around the inference logic located in `src/predict.py`. Always import and use `PredictionResult` and inference utilities from `src` instead of duplicating model loading.
- **Medical Disclaimer**: UI must prominently display that the tool is a research classification tool.
- **Confidence Flagging**: If the confidence score from inference is below 60%, the UI must automatically flag the result for human review.
- **Modularity**: Try keeping UI rendering separate from model operations.
