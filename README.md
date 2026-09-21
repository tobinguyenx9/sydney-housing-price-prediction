# Sydney Housing Price Prediction

SIT307 Task 8.1D. Predicting house prices in three Sydney suburbs
(Blacktown, Parramatta, Strathfield) from 118 sold listings collected
manually from Domain.

## Files

| File | Description |
|---|---|
| `analysis.ipynb` | Full analysis: data checks, exploration, feature engineering, three models, error analysis |
| `sydney_housing.xlsx` | Collected dataset (sheet `data`) and a 32-entry collection log (sheet `notes`) |
| `app.py` | Streamlit app that predicts a price and warns when the two models disagree |
| `housing_models.joblib` | Random Forest and Linear Regression trained on all 118 properties |

## Running the app

pip install streamlit scikit-learn pandas openpyxl joblib
streamlit run app.py


## Results

Random Forest gives the lower typical error (median 10.5% against 12.7%
for Linear Regression), but under-predicts properties above AUD 2.4M.
The app shows both models and flags large disagreements.
