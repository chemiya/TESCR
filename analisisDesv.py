import pandas as pd
import numpy as np


archivo_csv = 'CDs_and_Vinyl-100k.csv'

# Leer 
df = pd.read_csv(archivo_csv, names=["itemId","userId","rating","timestamp"])





columna_distinta = 'userId'


columna_desviacion_tipica = 'timestamp'

# Crear un DataFrame 
resultados = pd.DataFrame(columns=['Valor', 'DesviacionTipica','Cuenta'])

# recorro unicos
valores_unicos = df[columna_distinta].unique()
contador=0
total=len(valores_unicos)
for valor_unico in valores_unicos:
    contador=contador+1
    print("Llego por:",str(contador),"/",str(total))
    df_filtrado = df[df[columna_distinta] == valor_unico]
    desviacion_tipica = df_filtrado[columna_desviacion_tipica].std()
    
    if np.isnan(desviacion_tipica):
        desviacion_tipica=0
    
    
    nueva_fila = pd.Series({'Valor': valor_unico, 'DesviacionTipica': desviacion_tipica,'Cuenta':len(df_filtrado)})
    resultados.loc[len(resultados)] = nueva_fila

# excel
archivo_excel = 'resultadosDesv.xlsx'
resultados.to_excel(archivo_excel, index=False)

print(f'Se han guardado los resultados en "{archivo_excel}"')

