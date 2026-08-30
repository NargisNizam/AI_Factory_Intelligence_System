# AI Factory Intelligence Command Center
## Autonomous Manufacturing Intelligence & Digital Twin — AI Factory 2.0

An integrated, industry-style decision-support system combining:
- Tabular + time-series factory data
- Machine-learning predictive maintenance
- Deep-learning image defect classification
- NLP maintenance-note analysis
- RAG over factory SOP/manual knowledge
- Three cooperating AI agents
- SHAP-style/feature-importance explainability
- Digital-twin what-if simulation
- Human-in-the-loop approval
- MLflow experiment tracking
- Streamlit web application
- Automated PDF incident report

> Safety note: recommendations are decision support only and require human supervisor approval.

## Quick start
```bash
pip install -r requirements.txt
python src/generate_data.py
python src/train_models.py
streamlit run app.py
```

If TensorFlow is unavailable, the application still demonstrates the complete orchestration flow using the saved tabular model and deterministic image/NLP fallbacks.

## Project structure
- `app.py` — command center UI
- `src/generate_data.py` — synthetic multimodal factory data
- `src/train_models.py` — baseline ML + deep learning + MLflow
- `src/pipeline.py` — agents, RAG, XAI and digital twin
- `src/report.py` — PDF generation
- `data/` — datasets
- `knowledge_base/` — SOP/manual text
- `models/` — trained models
- `reports/` — generated reports

## Demonstrated modalities
1. Tabular production/machine records
2. Time-series sensor readings
3. Product images
4. Maintenance text
5. Machine SOP/manual knowledge

## Three meaningful experiments
MLflow tracks:
1. Logistic Regression baseline
2. Random Forest
3. Gradient Boosting

A neural network is additionally trained for the deep-learning requirement.

## Limitations
The factory data are synthetic and the image classifier uses a small synthetic visual dataset for demonstration. A real deployment would require validated industrial sensor data, real defect images, calibrated thresholds, secure model serving, monitoring, and domain-expert approval.
