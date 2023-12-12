import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# importing metrics

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

st.title("Model Creation/Evaluation")

# Load cleaned dataset
feature_columns = ['peso_avg', 'presion_avg', 'presion_max', 'visco_max']
# feature_columns = ['presion_max', 'visco_max']
target_columns = ['densidad_max']

df = pd.read_csv('cleaned_data.csv')
df = df[feature_columns + target_columns]

st.write(df.head())

st.markdown(f'Shape: {df.shape}')

# Setting X and y
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
N_FOLDS = 5

st.markdown(f'## Creating Regression Models')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)
rf_model = RandomForestRegressor(n_estimators=100, random_state=20)
ada_model = AdaBoostRegressor(n_estimators=100, random_state=20)
xgb_model = XGBRegressor(n_estimators = 100)


python_code = """
# Setting X and y
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
N_FOLDS = 5

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)
rf_model = RandomForestRegressor(n_estimators=100, random_state=22)

"""

# Display the code in Streamlit
st.code(python_code, language='python')

st.write(f'## Getting cross-validation score do validation')

python_code = """
cv_scores = cross_val_score(rf_model, X, y, cv=N_FOLDS)
"""
# Display the code in Streamlit
st.code(python_code, language='python')

# Calculate Scores:
cv_scores_rf = cross_val_score(rf_model, X, y, cv=N_FOLDS)
cv_scores_ada = cross_val_score(ada_model, X, y, cv=N_FOLDS)
cv_scores_xgb = cross_val_score(xgb_model, X, y, cv=N_FOLDS)

# DATA MODELS

# rf_data
df_model_rf = pd.DataFrame(
    {
        'cv_scores': cv_scores_rf
    }
)
df_model_rf['model'] = 'Random Forest'

# ada_data
df_model_ada = pd.DataFrame(
    {
        'cv_scores': cv_scores_ada
    }
)
df_model_ada['model'] = 'Ada Boost'

# xgb_data
df_model_xgb = pd.DataFrame(
    {
        'cv_scores': cv_scores_xgb
    }
)
df_model_xgb['model'] = 'XGBoost'


df_models = pd.concat([df_model_ada, df_model_rf, df_model_xgb])

fig = px.box(data_frame=df_models, y='cv_scores', color='model', template='seaborn')
fig.update_layout(
    title_text=f'Cross-Validation Scores among {N_FOLDS} folds',
)

st.plotly_chart(fig)

st.write(f'### The model seems to be suiting well as the diffence among various dataframes are not too big')

st.write(f'## Train the model')

python_code = """
rf_model.fit(X_train, y=y_train)
"""
# Display the code in Streamlit
st.code(python_code, language='python')

rf_model.fit(X_train, y=y_train)
ada_model.fit(X_train, y=y_train)
xgb_model.fit(X_train, y=y_train)

st.write(f'## Calculate Metrics')

python_code = """
y_pred = rf_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
"""
st.code(python_code, language='python')

# Predictions RF

y_pred_rf = rf_model.predict(X_test)

mse_rf = mean_squared_error(y_test, y_pred_rf)
mape_rf = mean_absolute_percentage_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

st.markdown(f'### Random Forest Predictions:')
st.write(f'MAPE (RF): {mape_rf:.2f}%')
st.write(f'MAE (RF): {mae_rf}')
st.write(f'MSE (RF): {mse_rf}')
st.write(f'R2 (RF): {r2_rf}')

# Predictions ADA

y_pred_ada = ada_model.predict(X_test)

mse_ada = mean_squared_error(y_test, y_pred_ada)
mape_ada = mean_absolute_percentage_error(y_test, y_pred_ada)
mae_ada = mean_absolute_error(y_test, y_pred_ada)
r2_ada = r2_score(y_test, y_pred_ada)

st.markdown(f'### Ada Boost Predictions:')
st.write(f'MAPE (ADA): {mape_ada:.2f}%')
st.write(f'MAE (ADA): {mae_ada}')
st.write(f'MSE (ADA): {mse_ada}')
st.write(f'R2 (ADA): {r2_ada}')

# Predictions XGB

y_pred_xgb = xgb_model.predict(X_test)

mse_xgb = mean_squared_error(y_test, y_pred_xgb)
mape_xgb = mean_absolute_percentage_error(y_test, y_pred_xgb)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

st.markdown(f'### XGBoost Predictions:')
st.write(f'MAPE (XGB): {mape_xgb:.2f}%')
st.write(f'MAE (XGB): {mae_xgb}')
st.write(f'MSE (XGB): {mse_xgb}')
st.write(f'R2 (XGB): {r2_xgb}')

# Create RF Error Metrics DF

df_error_metrics_rf = pd.DataFrame()

df_error_metrics_rf['y_test'] = y_test
df_error_metrics_rf['y_pred'] = y_pred_rf
df_error_metrics_rf['model'] = 'Random Forest'

# Create ada Error Metrics DF

df_error_metrics_ada = pd.DataFrame()

df_error_metrics_ada['y_test'] = y_test
df_error_metrics_ada['y_pred'] = y_pred_ada
df_error_metrics_ada['model'] = 'Ada Boost'

# Create xgb Error Metrics DF

df_error_metrics_xgb = pd.DataFrame()

df_error_metrics_xgb['y_test'] = y_test
df_error_metrics_xgb['y_pred'] = y_pred_xgb
df_error_metrics_xgb['model'] = 'XGBoost'

df_error_metrics = pd.concat([df_error_metrics_ada, df_error_metrics_rf, df_error_metrics_xgb])


# Criar scatter plot com linha de referência
fig = px.scatter(df_error_metrics, x='y_test', y='y_pred', color='model', labels={'y_test': 'Real Values', 'y_pred': 'Predicted Values'},
                 title=f'Scatter Plot', template='seaborn')

# Adicionar linha de referência y=x
reference_line = px.line(x=[min(y_test), max(y_test)], y=[min(y_test), max(y_test)],)
fig.add_trace(reference_line.data[0])


# Exibir o gráfico no Streamlit
st.plotly_chart(fig)

st.markdown(f'## Conclusion')

st.write("""
The XGBoost model exhibited a smaller range of scores during cross-validation, suggesting that it consistently performed well across different subsets of the training data. This indicates a higher level of stability and reliability in its predictions compared to other models that may have shown more variability.

Moreover, when evaluating the model's performance metrics, specifically Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE), the XGBoost model outperformed all other models. The smaller MAE indicates that, on average, the model's predictions were closer to the actual values, while the lower MAPE implies a better percentage accuracy in predicting the target variable.

Given these observations, the conclusion is that the XGBoost model not only adapted well to the training data but also demonstrated superior performance in terms of accuracy and stability. As a result, XGBoost stands out as the preferred or recommended model for addressing the specific problem at hand.
""")