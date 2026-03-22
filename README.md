# plant_disease_detection

Complete Streamlit website for plant disease prediction.

## What Works Now

- Website runs even if TensorFlow/model is not available (demo fallback mode).
- Uses trained model automatically when `plant_disease_model.keras` exists.
- Reads label names from `class_names.json`.

## Project Files

- `app.py`: main Streamlit website.
- `app.txt`: backup copy of app code.
- `requirements.txt`: required packages for running the website.
- `class_names.json`: default prediction class labels.
- `plant_disease.ipynb`: optional notebook for model training.

## Quick Start

1. Open terminal in project folder.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run website:

```bash
streamlit run app.py
```

4. Open browser URL shown by Streamlit (usually `http://localhost:8501`).

## Real Model Mode (Optional)

To use real TensorFlow predictions instead of demo fallback:

1. Install TensorFlow:

```bash
pip install tensorflow
```

2. Train/export your model to project root as:

- `plant_disease_model.keras`

3. Restart Streamlit app.

The app sidebar shows current mode:

- `Mode: Demo fallback`
- `Mode: Trained model`

## Public Deployment (Streamlit Community Cloud)

1. Open: `https://share.streamlit.io/`
2. Sign in with GitHub.
3. Click `New app` and choose:
- Repository: `venkydandi/plant_disease_detection`
- Branch: `main`
- Main file path: `app.py`
4. Click `Deploy`.

After deployment, Streamlit gives a public live link.