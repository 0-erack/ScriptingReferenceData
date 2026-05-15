import pandas as pd #Usando Pandas para la manipulacion de datos
import glob

df = pd.read_csv('Ultimate_Games_Dataset.csv') #Importar dataset, puede ser un csv (seria una tabla)
#Cuantitativo(discreto, count), nominal, ordinal
print(df.head(10)) #Mostrar algunos samples del dataset
print(df) #Mostrar dataset entero (no es un array, es un objeto especial de Pandas)
print(len(df)) #Cantidad de filas
print(df.dtypes) #Mostrar tipos asignados de las columnas (int32, int64, object, float32, float64, string, category)
df['release_year'].fillna(0, inplace = True) #Reemplaza los no presentes por 0, haria referencia a una columna
df['release_year'] = df['release_year'].astype("int64") #Cambiar el tipo de una columna
print(df['release_year'].unique()) #Mostrar solo los unicos
df['controls'] = pd.Categorical(df['controls'], ['Controller & Keyboard/Mouse', 'Keyboard / Mouse', 'Controller', 'Not Specified'], ordered=True) #Establece la columna como categorica, asi que sera como un enum y se ordenara asi
print(df['controls'].unique())
df = pd.get_dummies(data=df, columns=['view_dimension']) #OneHotEncoding, hara columnas por cada valor distinto poniendo False o True
df['nuevo'] = df['title'].add("--") #Crear nueva columna haciendo operaciones bulk
print(df.info()) #Info variada del dataframe
print(df.describe())
print(df.columns)
print(df['controls'].value_counts()) #Cantidad de cada valor presente
#df.to_html('output.html') #Guardar el dataset como tabla html




#Concatenar datos de multiples csv en el mismo dataframe
def ver_dataframes():
    files = glob.glob("*.csv")
    df_list = []
    for filename in files:
        data = pd.read_csv(filename)
        df_list.append(data)
    df = pd.concat(df_list)
    return df
