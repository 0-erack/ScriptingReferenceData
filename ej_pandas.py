import pandas as pd #Usando Pandas para la manipulacion de datos
import glob
import seaborn as sns
import matplotlib.pyplot as plt

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
print(df.duplicated()) #Lista los id cuyos valores estan enteramente duplicados, df.duplicated().value_counts() devolveria la cantidad de duplicados y no duplicados

df['nuevo'] = df['title'].add(' y ya') #Crear nueva columna haciendo operaciones bulk
df['inicial'] = df.title.str[0:1] #Dividir string
separado = df['title'].str.split('-') #Separar un valor en dos o mas (cantidad fija)
df['titulo1'] = separado.str.get(0)
df['titulo2'] = separado.str.get(1)
df['developers'] = df['developers'].str.split('|') #Separar a|b|c en a, b, c (columnas)
df = df.explode('developers') #Y luego en valores
df['view_dimension'] = df['view_dimension'].replace('D', 'd', regex=True) #Reemplazar valores
df['dimension'] = df['view_dimension'].str.split('(\d+)', expand=True)[0] #Recogiendo solo el numero
df['dimension'] = df['view_dimension'].str.extract('(\d+)') #Mas sencillo que split

df = df.drop('status_yet', axis=1) #Borrar una columna
df['controls'] = pd.Categorical(df['controls'], ['Controller & Keyboard/Mouse', 'Keyboard / Mouse', 'Controller', 'Not Specified'], ordered=True) #Establece la columna como categorica, asi que sera como un enum y se ordenara asi
recortado = df[['title', 'game_id']] #Recortar columnas
df = pd.get_dummies(data=df, columns=['view_dimension']) #OneHotEncoding, hara columnas por cada valor distinto poniendo False o True
df['release_year'] = df['release_year'].fillna(0) #Reemplaza los no presentes por 0, haria referencia a una columna
df = df.dropna(subset=['release_year']) #Directamente borra los registros con datos n/a en esa columna, puede no recibir argumentos
df['release_year'] = df['release_year'].astype('int64') #Cambiar el tipo de una columna, lowerCamelCase no admite nulos pero HigherCamelCase (Int) si
df['release_year'] = pd.to_numeric(df['release_year']) #Convertir a numerico mas flexible
separado = pd.melt(frame=df, id_vars=['title'], value_vars=['metacritic', 'ratings_count'], value_name='rating', var_name='rater') #Crear un dataframe separando medidas de x columnas en x filas, se preservaria title (osea por cada title) y rater tendria 'metacritic' o 'ratings_count' y su valor estaria en rating
df = df.drop_duplicates(subset=['title']) #Borra los que tengan ese valor duplicado, quedandose con el primero, puede no recibir nada
df['avg_playtime_hours'] = df['avg_playtime_hours'].ffill(axis=0) #Intenta rellenar los huecos con aproximaciones, util para numericos o booleanos (locf para de antes a despues, para de despues a antes seria nocb y se usaria bfill, tambien estan bocf y wocf)
df['title'] = df['title'].apply(lambda x: x + ": el juego") #Aplicar una transformacion lambda en todos los valores
subset = df.loc[(df['release_year'] > 2000) & (df['platform_count'] > 1)] #Recojer una muestra del dataset
subset = df.iloc[10:21, 0:3] #Igual pero por indices
stats_por_agno = df.groupby('release_year')['popularity_score'].mean().sort_values(ascending=False) #Agrupar por datos (por agno), poner un valor con agregacion (media de popularidad), y ordenar
subset = df[df['theme'] == 'Fantasy'].sort_values(by='title') #Filtrar y ordenar
completo = pd.merge(df, subset, on='game_id', how='left') #Nuevos dataframes con joins
matriz = df.pivot_table(values='popularity_score', index='release_year', columns='all_genres', aggfunc='mean') #Parecido a groupby pero con matriz

#df.to_html('output.html') #Guardar el dataset como tabla html
#df.to_csv('limpios.csv', index=False) #Exportar a csv
#Visualizar estos datos
#sns.scatterplot(data=pd.read_csv('Ultimate_Games_Dataset.csv'), x='platform_count', y='metacritic', hue='game_mode')
#plt.show()

#Concatenar datos de multiples csv en el mismo dataframe
def ver_dataframes():
    files = glob.glob('*.csv')
    df_list = []
    for filename in files:
        data = pd.read_csv(filename)
        df_list.append(data)
    df = pd.concat(df_list, ignore_index=True)
    return df
