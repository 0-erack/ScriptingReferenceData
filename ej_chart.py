import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt #Matplotlib es una libreria para dibujar datos y graficas compatible con datasets de Pandas
import seaborn as sns #Seaborn actua sobre Matplotlib haciendo mas faciles algunas cosas y extendiendo funcionalidades
import squarify

df = pd.read_csv('min.csv') #Se usa un dataframe de Pandas
print(df.head())
print(df.dtypes)
sns.set_theme(style="whitegrid") #Establecer un tema

plt.figure(figsize=(6,8)) #Ventana basica
plt.show() #Mostrar grafico actual y limpiar el buffer

a=sns.catplot(data=df, x="decade", y="popularity_score", kind="bar") #Declarar un grafico con Seaborn, en este caso de barras
#plt.savefig('grafico_popularidad.png', dpi=300, bbox_inches='tight') #Guardar el grafico actual en el buffer
plt.show()

df_agrupado = df.groupby('decade')['popularity_score'].mean().reset_index()
plt.bar(df_agrupado['decade'], df_agrupado['popularity_score']) #Lo mismo pero con Matplotlib (normalmente es mas comodo Seaborn)
plt.show()

sns.histplot(data=df, x='metacritic', bins=20, kde=True, color='royalblue') #Histograma
plt.title("Grafico", fontsize=12, fontweight='bold') #Estilizado
plt.xlabel('Puntuacion')
plt.ylabel('Cantidad')
plt.show()

sns.scatterplot(data=df, x='avg_playtime_hours', y='metacritic', hue='game_mode', alpha=0.7, size='platform_count') #Grafico de dispersion
plt.show()

sns.lineplot(data=df, x="release_year", y="popularity_score") #Lineas
plt.xlim(1980, 2026) #Recortar
plt.xticks(rotation=45)
plt.show()

sns.boxplot(data=df, x='view_dimension', y='user_rating', palette='Pastel1') #Cajas
plt.show()

sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1) #Mapa de calor, en este caso para encontrar correlacion entre las columnas numericas
plt.tight_layout()
plt.show()

data_pie = df['view_dimension'].value_counts()
plt.pie(data_pie, labels=data_pie.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('Pastel1'), wedgeprops={'edgecolor': 'white', 'linewidth': 2}) #Grafico de tarta
plt.show()

pd.crosstab(df['release_year'], df['view_dimension']).plot(kind='bar', stacked=True, color=sns.color_palette('Set2'), ax=plt.gca()) #Barras apiladas
plt.show()

data_tree = df['controls'].value_counts().head(10)
squarify.plot(sizes=data_tree.values, label=data_tree.index, alpha=0.8, color=sns.color_palette('viridis', len(data_tree))) #Cajas de tamagnos, usa Squarify
plt.show()

#Se pueden lograr mas tipos de graficos como de sankey, waterfall, radar, mapas, de calor, etc.
