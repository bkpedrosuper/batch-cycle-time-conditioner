import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Data Analysis")


# read csv
df = pd.read_csv('FC_Mixer_MO.csv')

st.markdown('## Check the first 10 register to take a grasp of what the dataset look like')
st.write(df.head())

st.markdown('## Checking how the number features are distributed')
st.write(df.describe())

st.markdown('## Check features types')
st.write(df.dtypes)

st.markdown('## Now, lets remove every column that has the same value in all dataset')
st.markdown('Check which columns have different values')

columns_to_drop = []
for col in df.columns:
    unique = df[col].unique().shape[0]
    if unique == 1:
        
        columns_to_drop.append(col)

st.write(f'Columns to drop: {columns_to_drop}')
df = df.drop(columns=columns_to_drop)

# st.write('## Check how the features are distributed')

# st.write('### Variables that may exhibit greater significance for the problem are sought')
# selected_columns = df.columns[:1]
# for col in selected_columns:
#     # Adjust matplotlib main settings
#     fig = px.histogram(data_frame=df, x=col)
#     fig.update_layout(
#         title_text=f'Histograma - {col}',
#         xaxis_title_text='Valores',
#         yaxis_title_text='Contagem'
#     )

#     # plot figure
#     st.plotly_chart(fig)

st.write('## Check how the targets are distributed')
output_variables = ['char_value']
for var in output_variables:

    fig = px.histogram(data_frame=df, x=var)
    fig.update_layout(
        title_text=f'Histograma - {var}',
        xaxis_title_text='Valores',
        yaxis_title_text='Contagem'
    )
    st.plotly_chart(fig)
    
    fig = px.box(data_frame=df, y=var)
    fig.update_layout(
        title_text=f'Boxplot - {var}',
    )
    st.plotly_chart(fig)

st.write('### Check the distribution between char_name and char_value')
fig = px.bar(data_frame=df, x='char_name', y='char_value')
fig.update_layout(
        title_text=f'Distribuição - char_name x char_value',
    )
st.plotly_chart(fig)


st.markdown("""
## Estrategy adopted:
Unite COMPLETE batches with the same uniqueid into a "measure". Each uniqueid will have N measures, being N the number of phases recorded at phase 0 for each uniqueid.

- Remove category columns that has 1 unique value ['site', 'division', 'category', 'line', 'model_name', 'recipe_group', 'product_code']
- Rename useful columns
- Remove INCOMPLETE measures
- Transform each uniqueid in N measures
- Set the Densidad_max variable to target
""")