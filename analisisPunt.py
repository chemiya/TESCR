import pandas as pd
import numpy as np


archivo_csv = 'CDs_and_Vinyl-100k.csv'

# Leer 
df = pd.read_csv(archivo_csv, names=["itemId","userId","rating","timestamp"])





columna_distinta = 'userId'


columna_desviacion_tipica = 'rating'

# Crear un DataFrame 
resultados = pd.DataFrame(columns=['Valor', 'Rating0', 'Rating1', 'Rating2', 'Rating3', 'Rating4', 'Rating5','Cuenta'])

# recorro unicos
valores_unicos = df[columna_distinta].unique()
contador=0
total=len(valores_unicos)
for valor_unico in valores_unicos:
    contador=contador+1
    print("Llego por:",str(contador),"/",str(total))
    
    df_filtrado = df[df[columna_distinta] == valor_unico]
    tamano=len(df_filtrado)
    contador_filas_valor_0 = round((len(df_filtrado[df_filtrado['rating'] == 0])/tamano)*100,2)
    contador_filas_valor_1 = round((len(df_filtrado[df_filtrado['rating'] == 1])/tamano)*100,2)
    contador_filas_valor_2 = round((len(df_filtrado[df_filtrado['rating'] == 2])/tamano)*100,2)
    contador_filas_valor_3 = round((len(df_filtrado[df_filtrado['rating'] == 3])/tamano)*100,2)
    contador_filas_valor_4 = round((len(df_filtrado[df_filtrado['rating'] == 4])/tamano)*100,2)
    contador_filas_valor_5 = round((len(df_filtrado[df_filtrado['rating'] == 5])/tamano)*100,2)
    

    
    
    nueva_fila = pd.Series({'Valor': valor_unico, 'Rating0':contador_filas_valor_0, 'Rating1':contador_filas_valor_1, 'Rating2':contador_filas_valor_2, 'Rating3':contador_filas_valor_3, 'Rating4':contador_filas_valor_4, 'Rating5':contador_filas_valor_5,'Cuenta':tamano})
    resultados.loc[len(resultados)] = nueva_fila

# excel
archivo_excel = 'resultadosPunt.xlsx'
resultados.to_excel(archivo_excel, index=False)

print(f'Se han guardado los resultados en "{archivo_excel}"')

