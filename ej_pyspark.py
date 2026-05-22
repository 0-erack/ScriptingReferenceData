#Inicializar Spark, evolucion de Hadoop/HDFS y usado para manejar big data
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
#spark = SparkSession.builder.getOrCreate()
spark = SparkSession.builder.master('local[*]').config('spark.driver.memory', '1g').config('spark.app.name', 'ejemplo').getOrCreate() #Integracion SQL en Spark con dataframes
print(spark)

#Cargando datos paralelizados
data = [("aa", 1, "A"), ("bb", 2, "B"), ("cc", 3, "C"), ("dd", 4, "D"), ("ee", 5, "E")]
rdd = spark.sparkContext.parallelize(data, 3)
rdd.collect()

rdd_juegos = spark.sparkContext.parallelize([("1", "aaa"), ("2", "bbb"), ("3", "ccc")])
rdd_ventas = spark.sparkContext.parallelize([("1", 500), ("2", 250), ("3", 300)])
rdd_resultado = rdd_juegos.join(rdd_ventas) #Uniendo RDD si las clave-valor coinciden
print(rdd_resultado.collect())


#Cargando datos de un archivo (en este caso sin Hadoop)
#df_spark = spark.read.csv("min.csv", header=True, inferSchema=True)
#df = spark.read.parquet('./dataset/') #Leer Parquet (preparado para big data y distribucion)

#Cargando datos de un archivo
rdd = spark.sparkContext.textFile("min.csv", 15) #Paraleliza en 15 segmentos
rdd.collect() #No es necesario el collect si no queremos todos los datos
print(rdd.count())
print(rdd.getNumPartitions())
rdd = rdd.map(lambda x: x.split(',')).filter(lambda x: x[0].isdigit()) #LazyTransform
rdd_transformado = rdd.map(lambda x: x[2].upper())
contador = spark.sparkContext.accumulator(0) #Contadores para iteraciones de Spark
for i in rdd_transformado.take(5): #Iterar datos (justo asi no seria distribuido)
    contador.add(1)
rdd_transformado.foreach(lambda x: contador.add(1)) #Asi mejor
print(str(contador))
rdd_transformado = rdd.filter(lambda x: x[0].isdigit()).map(lambda x: int(x[0])).reduce(lambda x,y: x + y) #Usar reduce con operaciones con propiedad conmutativa como el + o el * para tener siempre el mismo resultado indepentientemente de como se paralelice
print(rdd_transformado)

broadcast = spark.sparkContext.broadcast({"3D": "tres", "2D": "dos"}) #A todos los nodos, no se distribulle, se copia
print(rdd.map(lambda x: broadcast.value.get(x[8], "Desconocido")).take(5)) #Aplicar cambios en base al broadcast, se podria unir con .join

rdd_df = rdd.map(lambda x: (x[0], x[1], x[2])).toDF(['serial', 'id', 'title']) #Convertir RDD a dataframe
rdd_df.show(5, truncate=20)
print(rdd_df.rdd.take(5)) #Y de dataframe a RDD
df = spark.read.option('header', True).option('delimiter', ',').option('inferSchema', True).csv("min.csv") #Dataframe en Spark directamente con un csv
df = df.withColumnRenamed("all_genres", "genres").drop("genres")
df2 = df.withColumn("game_id", df['game_id'] + 10) #Operaciones mas eficientes que las lambda
df2 = df.na.fill({"game_mode": 'NA'})
print(df.head())
df.printSchema()
print(df.count())
df.describe().show(truncate=15)
df.filter(df.title.contains("The")).select(['title']).orderBy('title', ascending=False).show(truncate=25) #Parecido a SQL, tambien hay agregacion por ejemplo .sum()
df.groupBy("release_year").agg(F.avg("achievements_count")).show(5)
#df.write.csv('./limpio/', mode="overwrite") #No es un guardado convencional, se almacena con partes preparado para Spark
#df.write.parquet('./limpio/', mode="overwrite")
df_filtrado = df.filter(df.metacritic > 4.0).cache() #Almacenar en memoria

df.createOrReplaceTempView('dataframe')
spark.sql("select * from dataframe;").show(10, truncate=30) #SQL directamente, antes hay que hacer la view
print(df_filtrado.show(5))

spark.stop()