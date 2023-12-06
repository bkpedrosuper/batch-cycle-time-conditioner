import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# importing metrics

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

st.title("Data Creation/Evaluation")

# Load cleaned dataset
feature_columns = ['peso_avg', 'presion_avg', 'presion_max', 'visco_max']
target_columns = ['densidad_max']

df = pd.read_csv('cleaned_data.csv')
df = df[feature_columns + target_columns]

st.write(df.head())

st.markdown(f'Shape: {df.shape}')

# Setting X and y
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
N_FOLDS = 10

st.markdown(f'## Creating Random Forest Regression Model')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)
rf_model = RandomForestRegressor(n_estimators=100, random_state=22)



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

cv_scores = cross_val_score(rf_model, X, y, cv=N_FOLDS)

st.write(f'CV Mean: {cv_scores.mean()}')
st.write(f'CV STD: {cv_scores.std()}')

fig = px.box(y=cv_scores)
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

st.write(f'## Calculate Metrics')

python_code = """
y_pred = rf_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
"""
st.code(python_code, language='python')

y_pred = rf_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f'MAPE: {mape:.2f}%')
st.write(f'MAE: {mae}')
st.write(f'MSE: {mse}')
st.write(f'R2: {r2}')

df_error_metrics = pd.DataFrame()

df_error_metrics['y_test'] = y_test
df_error_metrics['y_pred'] = y_pred


# Criar scatter plot com linha de referência
fig = px.scatter(df_error_metrics, x='y_test', y='y_pred', labels={'x': 'Valores Reais', 'y': 'Valores Previstos'},
                 title=f'Scatter Plot - MAE: {mae:.2f}, MSE: {mse:.2f}')

# Adicionar linha de referência y=x
reference_line = px.line(x=[min(y_test), max(y_test)], y=[min(y_test), max(y_test)],)
fig.add_trace(reference_line.data[0])
fig.update_traces(marker=dict(color='green'))


# Exibir o gráfico no Streamlit
st.plotly_chart(fig)
