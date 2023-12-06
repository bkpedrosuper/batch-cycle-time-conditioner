import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go


st.title("Data Cleaning")

# read csv
df = pd.read_csv('FC_Mixer_MO.csv')

# First of all, let's remove every column that has the same value for all dataset
# Check which columns have different values

columns_to_drop = []
for col in df.columns:
    unique = df[col].unique().shape[0]
    if unique == 1:
        
        columns_to_drop.append(col)

df = df.drop(columns=columns_to_drop)
# st.write(f'Columns to drop: {columns_to_drop}')

assert all(len(df[col].unique()) > 1 for col in df.columns)

# Rename useful columns
cols = ['id', 'complete', 'phase', 'time', 'value']
df = df.rename(columns={
    'uniqueid': 'id',
    'model_complete': 'complete',
    'char_name': 'phase',
    'char_time': 'time',
    'char_value': 'value'
})

# Remove INCOMPLETE measures

before = df.shape[0]
df = df[df['complete'] == 'COMPLETE']
after = df.shape[0]
removed_incomplete = before - after

assert len(df['complete'].unique()) == 1


st.write(df)
# Change phase names

df = df.replace('non_value_operating_time', 'phase_0')
df = df.replace('phase_1.Inicio.duration', 'phase_1')
df = df.replace('phase_2.Medicion.duration', 'phase_2')
df = df.replace('phase_3.Recirculacion.duration', 'duration')
df = df.replace('phase_3.Recirculacion.Peso_avg', 'peso_avg')
df = df.replace('phase_3.Recirculacion.Presion_avg', 'presion_avg')
df = df.replace('phase_3.Recirculacion.Presion_max', 'presion_max')
df = df.replace('phase_3.Recirculacion.Visco_max', 'visco_max')
df = df.replace('phase_3.Recirculacion.Densidad_max', 'densidad_max')
df = df.replace('total_duration', 'phase_global')

assert all(val != 'non_value_operating_time' for val in df['phase'])



st.markdown(f"""
Changes made:
            
- Remove category columns that has 1 unique value | {len(columns_to_drop)} columns
- Remove measures with complete value == INCOMPLETE | {before - after} rows
- Change column names:
    - uniqueid -> id
    - model_complete -> complete
    - char_name -> phase
    - char_time -> time
    - char_value -> value
- Change phase names:
    - non_value_operating_time -> phase_0
    - phase_1.Inicio.duration -> phase_1
    - phase_2.Medicion.duration -> phase_2
    - phase_3.Recirculacion.duration -> duration
    - phase_3.Recirculacion.Peso_avg -> peso_avg
    - phase_3.Recirculacion.Presion_avg -> presion_avg
    - phase_3.Recirculacion.Presion_max -> presion_max
    - phase_3.Recirculacion.Visco_max -> visco_max
    - phase_3.Recirculacion.Densidad_max -> densidad_max
- Unite COMPLETE batches with the same uniqueid into a "measure". Each uniqueid will have N measures, being N the number of phases recorded at phase 0 for each uniqueid.
    - Transform each uniqueid in N measures
    - Set the Densidad_max variable to target
""")

# Create cleaned dataframe

cleaned_columns = ['phase_0', 'phase_1', 'phase_2', 'duration', 'peso_avg', 'presion_avg', 'presion_max', 'visco_max', 'phase_global', 'densidad_max']

df_cleaned = pd.DataFrame(columns=cleaned_columns)

# Transform each id in N measures
for id in df['id'].unique():
    cleaned = pd.DataFrame()
    df_unique = df[df['id'] == id]

    # Get the number of measures for each id
    n_measures = df_unique[df_unique["phase"] == "phase_0"].shape[0]


    # Transform each value of phase into a column
    for col in cleaned_columns:
        # Get the N first values of the measure
        np_values = df_unique[df_unique['phase']==col]['value'].head(n_measures)
        
        # Transform into a list
        values = list(np_values)

        # Insert into the dataframe
        try:
            cleaned[col] = values
        except ValueError:
            cleaned[col] = np.concatenate([values, [np.nan] * (len(cleaned) - len(values))])

    # Concat with the main dataframe
    df_cleaned = pd.concat([df_cleaned, cleaned], ignore_index=True)

# Create duration_min column
df_cleaned['duration_min'] = df_cleaned['duration'] / 60

# Set order of the columns
order = ['phase_0', 'phase_1', 'phase_2', 'duration', 'duration_min', 'peso_avg', 'presion_avg', 'presion_max', 'visco_max', 'phase_global', 'densidad_max']
df_cleaned = df_cleaned[order]

st.markdown(f'## Behold The new Dataset! Shape: {df_cleaned.shape}')

st.write(df_cleaned)

st.markdown(f'## Distribuição das variáveis')


# Definir o número de linhas e colunas no grid
rows, cols = 6, 2

# Criar um grid de boxplots usando Plotly Express
fig = make_subplots(rows=rows, cols=cols, subplot_titles=df_cleaned.columns)


cc = 0
for i in range(1, rows + 1):
    for j in range(1, cols + 1):
        if cc >= len(df_cleaned.columns):
            break
        box_fig = px.box(df_cleaned, y=df_cleaned.columns[cc], title=f'BoxPlots for {df_cleaned.columns[cc]}')
        fig.update_yaxes(title_text='Values', row=i, col=j)
        fig.add_trace(go.Box(box_fig['data'][0]), row=i, col=j)

        cc += 1

# Atualizar layout com título geral
fig.update_layout(
    title_text='BoxPlots Grids',
    width=500, height=2000, 
)

# Exibir o gráfico no Streamlit
st.plotly_chart(fig)

st.markdown(f'## Cleaning Data')

# Remove NaN Values

before = df_cleaned.shape[0]
df_cleaned = df_cleaned.dropna()
after = df_cleaned.shape[0]
remove_nan = before - after

# Remove duplicate values

before = df_cleaned.shape[0]
df_cleaned = df_cleaned.drop_duplicates(subset=['peso_avg', 'presion_avg', 'presion_max', 'visco_max'])
after = df_cleaned.shape[0]
remove_duplicates = before - after

# Repair visco_max

# High values
percentile_95 = np.percentile(df_cleaned['visco_max'], 95)
df_cleaned['visco_max'] = np.where(df_cleaned['visco_max'] > 8000, percentile_95, df_cleaned['visco_max'] )

# Low or 0 values
percentile_50 = np.percentile(df_cleaned['visco_max'], 50)
df_cleaned['visco_max'] = np.where(df_cleaned['visco_max'] <= 0, percentile_50, df_cleaned['visco_max'] )


# Repair densidad_max

# Low or 0 values
percentile_5 = np.percentile(df_cleaned['densidad_max'], 5)
df_cleaned['densidad_max'] = np.where(df_cleaned['densidad_max'] <= 0, percentile_5, df_cleaned['densidad_max'] )

df_cleaned = df_cleaned[df_cleaned['densidad_max'] > 5000]

st.markdown(f"""
Changes made:
            
- Remove NaN Values | {remove_nan} rows
- Remove duplicated values | {remove_duplicates} rows
- Repair visco_max
- Repair densidad_max
- Create duration in minutes column
""")

st.markdown(f'## Correlation Test')

correlation_matrix = df_cleaned.corr()

# Criar um subplot
fig = make_subplots(
    rows=1, cols=1,
    subplot_titles=['Heatmap de Correlação']
)

fig = px.imshow(df_cleaned.corr(), text_auto=True, aspect='auto')

# Atualizar layout
fig.update_layout(title_text='Heatmap de Correlação', width=800, height=600)

# Exibir o gráfico no Streamlit
st.plotly_chart(fig)

st.markdown("""
Now with the dataset defined, it is possible to create a regression model for it

input features that have a strong enough correlation to train the model:
- peso_avg
- presion_avg
- presion_max
- visco_max
""")

st.markdown(f'## Final Dataset:')

st.write(df_cleaned)

st.markdown(f'Shape: {df_cleaned.shape}')

# Save to csv
df_cleaned.to_csv(f'cleaned_data.csv')

# Save to xlsx
df_cleaned.to_excel(f'cleaned_data.xlsx')

st.markdown(f'Final dataset shape: {df_cleaned.shape}')