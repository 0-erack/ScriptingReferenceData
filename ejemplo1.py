import pandas as pd #Usando Pandas para la manipulacion de datos

dataset = pd.read_csv('Ultimate_Games_Dataset.csv') #Importar dataset, puede ser un csv (seria una tabla)
print(dataset.head()) #Mostrar algunos samples del dataset
print(dataset) #Mostrar dataset entero (no es un array, es un objeto especial de Pandas)
print(dataset.dtypes) #Mostrar tipos asignados de las columnas (int32, int64, object, float32, float64, string)
dataset['release_year'].fillna(0, inplace = True) #Reemplaza los no presentes por 0, haria referencia a una columna
dataset['release_year'] = dataset['release_year'].astype("int64") #Cambiar el tipo de una columna
print(dataset['release_year'].unique()) #Mostrar solo los unicos