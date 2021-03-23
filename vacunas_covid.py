import streamlit as st
import pandas as pd
#import base64
#import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.title('Personas Vacunadas del Covid-19 en Sudamérica ')
st.markdown("""
Esta aplicación muestra las estadísticas de personas vacunadas!
* **Librerías de python usadas:** pandas, streamlit
* **Fuente de Información:** [www.kaggle.com](https://www.kaggle.com/gpreda/covid-world-vaccination-progress).
""")

st.sidebar.header('Clasificar por País')


#leyendo el archivo que tiene toda la informacion
df = pd.read_csv("https://docs.google.com/spreadsheets/d/1MlLL19lch5UGJkTtiiKf_yMDJRhzBpfF4ngSOuFVzPk/export?format=csv")

#limpiando los datos
df_cl = df.fillna(0)

#clasificando la informacion
data_fr = df_cl[(df_cl["total_vaccinations"] != 0)]

#seleccionando paises de Latam
data_fr_ = data_fr[(data_fr["country"].str.contains("Ecuador")) |
                   (data_fr["country"].str.contains("Colombia")) | 
                   (data_fr["country"].str.contains("Chile")) | 
                   (data_fr["country"].str.contains("Argentina")) | 
                   (data_fr["country"].str.contains("Brazil")) |
                   (data_fr["country"].str.contains("Bolivia")) |
                   (data_fr["country"].str.contains("Paraguay")) | 
                   (data_fr["country"].str.contains("Venezuela")) |
                   (data_fr["country"].str.contains("Uruguay")) |
                   (data_fr["country"].str.contains("Peru"))]

# Barra Lateral - Seleccionar Pais
pais_unico = sorted(data_fr_.country.unique())
seleccion_pais = st.sidebar.multiselect('Pais', pais_unico, pais_unico)

# Filtrando datos
df_seleccion_pais = data_fr_[(data_fr_.country.isin(seleccion_pais))] # & (data_fr_.Pos.isin(seleccion_mes))

st.header('Mostrar información del país seleccionado')
st.write('Dimensión de Datos: ' + str(df_seleccion_pais.shape[0]) + ' filas y ' + str(df_seleccion_pais.shape[1]) + ' columnas.')
st.dataframe(df_seleccion_pais)

#st.line_chart(data_fr_)   dato visualizable

#Vacunas que se estan usando en latam
vacuna2 = df_seleccion_pais.vaccines.value_counts()
#vacuna2.plot.pie()

#st.pyplot(vacuna2) dato no visualizable
#st.plotly_chart(df_seleccion_pais.vaccines)

#personas vacunadas
#total = df_seleccion_pais.people_vaccinated.sum()
st.write('Tipos de vacunas que estan usando en la region')
vacuna = pd.DataFrame(df_seleccion_pais.vaccines.value_counts())
st.bar_chart(vacuna)

#st.pyplot(plt.plot(df_seleccion_pais.vaccines))


if st.button('Total de Personas Vacunadas'):
    primer_dosis = (df_seleccion_pais.people_vaccinated.sum())
    st.write('Personas que recibieron la primera dosis son: '+str(primer_dosis))
    segunda_dosis =(df_seleccion_pais.people_fully_vaccinated.sum())
    st.write('Personas que recibieron la segunda dosis son: '+str(segunda_dosis))
    total_vacunados =(df_seleccion_pais.total_vaccinations.sum())
    st.write('Total de personas vacunadas en la region: '+str(total_vacunados))


# Heatmap
if st.button('Comparativa entre paises'):
    st.header('Información general ')
       #df_seleccion_pais.to_csv('output.csv',index=False)
       #df_e = pd.read_csv('output.csv')

    corr = df_seleccion_pais.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(7, 5))
        ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
        ax = sns.heatmap(corr, vmax=1, square=True)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
