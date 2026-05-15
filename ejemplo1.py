import pandas as pd #Usando Pandas para la manipulacion de datos
import glob

df = pd.read_csv('Ultimate_Games_Dataset.csv') #Importar dataset, puede ser un csv (seria una tabla)
#Cuantitativo(discreto, count), nominal, ordinal
dataframe2 = pd.DataFrame({'id': [0,1,2], 'letra': ['a', 'b', 'c']}) #Crear uno nuevo

print(df.head(10)) #Mostrar algunos samples del dataset
print(df) #Mostrar dataset entero (no es un array, es un objeto especial de Pandas)
print(len(df)) #Cantidad de filas
print(df['title'].head()) #Operaciones tambien con columnas
print(df.dtypes) #Mostrar tipos asignados de las columnas (int32, int64, object, float32, float64, string, category)

print(df['release_year'].unique()) #Mostrar solo los unicos
print(df.info()) #Info variada del dataframe
print(df.describe())
print(df.columns)
print(df['controls'].value_counts()) #Cantidad de cada valor presente
print(df['platform_count'].mean()) #La media
print(df.platform_count.mean()) #En algunos casos las columnas tambien se pueden representar de esta manera

df['nuevo'] = df['title'].add(' y ya') #Crear nueva columna haciendo operaciones bulk
df['inicial'] = df.title.str[0:1] #Dividir string
separado = df['title'].str.split('-') #Separar un valor en dos o mas (cantidad fija)
df['titulo1'] = separado.str.get(0)
df['titulo2'] = separado.str.get(1)
separado2 = df['developers'].str.split('|') #Separar a|b|c en a, b, c (columnas)
separado2 = separado2.explode('developers') #Y luego en valores

df['controls'] = pd.Categorical(df['controls'], ['Controller & Keyboard/Mouse', 'Keyboard / Mouse', 'Controller', 'Not Specified'], ordered=True) #Establece la columna como categorica, asi que sera como un enum y se ordenara asi
recortado = df[['title', 'game_id']] #Recortar columnas
df = pd.get_dummies(data=df, columns=['view_dimension']) #OneHotEncoding, hara columnas por cada valor distinto poniendo False o True
df['release_year'].fillna(0, inplace = True) #Reemplaza los no presentes por 0, haria referencia a una columna
df['release_year'] = df['release_year'].astype('int64') #Cambiar el tipo de una columna
#df.to_html('output.html') #Guardar el dataset como tabla html
separado = pd.melt(frame=df, id_vars=['title'], value_vars=['metacritic', 'ratings_count'], value_name='rating', var_name='rater') #Crear un dataframe separando medidas de x columnas en x filas, se preservaria title (osea por cada title) y rater tendria 'metacritic' o 'ratings_count' y su valor estaria en rating
print(df.duplicated()) #Lista los id cuyos valores estan enteramente duplicados, df.duplicated().value_counts() devolveria la cantidad de duplicados y no duplicados
df = df.drop_duplicates(subset=['title']) #Borra los que tengan ese valor duplicado, quedandose con el primero, puede no recibir nada


#Concatenar datos de multiples csv en el mismo dataframe
def ver_dataframes():
    files = glob.glob('*.csv')
    df_list = []
    for filename in files:
        data = pd.read_csv(filename)
        df_list.append(data)
    df = pd.concat(df_list)
    return df
