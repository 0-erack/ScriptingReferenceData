import numpy as np #Base optimizada del ecosistema de datos en Python

dimension1 = np.array([0,1,2,3]) #Vector
dimension2 = np.array([[0,1,2,3], [4,5,6,7]]) #Matriz
dimension3 = np.array([[[0,1,2,3], [4,5,6,7]], [[8,9,10,11], [12,13,14,15]]]) #Cubo
print(dimension3.ndim) #Cantidad de dimensiones
print(dimension2.shape) #Tamagno en cada eje
print(dimension2.dtype) #Tipo de dato
print(dimension1) #Datos talcual

#eje = np.array([[3,"fg",6], [7]]) #Debe tener el mismo tamagno pero puede tener tipos de datos distintos
print(np.array([3,"g",6, False]).dtype) #Igualmente no es recomendable, siempre es mejor arrays homogeneos
print(np.zeros((2,3))) #Matriz 2x3 lleno de 0s, tambien hay np.ones
print(np.arange(0, 10, 2)) #Del 0 al 10 en saltos de 2
print(np.linspace(0, 1, 5)) #Del 0 al 1, 5 numeros (float)
print(np.random.hypergeometric(ngood=5,nbad=5,nsample=6,size=100)) #Valores aleatorios

print(dimension3[1,0,2]) #Elemento concreto
print(dimension2[1, :]) #Fila 1
print(dimension1 * 2) #Calculos a todos los elementos
print(dimension2[:, 0] ** 2)
print(dimension1 + dimension1) #Itera y opera cada elemento

print(np.sum(dimension2)) #Suma de todos los elementos
print(np.average(dimension3)) #La media total, hay mas funciones de agregacion
print(np.sum(dimension2, axis=0)) #Suma hacia un eje concreto, de esta manera no devolvera un solo valor (0 columnas, 1 filas, etc)
print(np.std(dimension3)) #Desviacion estandar global
print(np.argmax(dimension2)) #Numero maximo, tambien esta argmin
print(np.cumsum(dimension2, axis=1)) #Suma acumulativa
print(np.percentile(dimension1, 50)) #Calculo de percentil

filtro = dimension1 > 2 #True para los que cumplan la condicion, devuelve un array de las mismas propiedades
print(filtro) 
print(dimension1[filtro]) #Solo las que respectivamente sean true
print(np.where(dimension1 > 1, "mas", "menos")) #Valor custom en lugar de booleano
nuevo = dimension2.reshape(1, 8) #Redimensionar array, el contenido es igual, debe terminar teniendo el mismo numero de elementos y se pueden poner o quitar dimensiones
print(dimension2.T) #Girar array
print(dimension2.flatten()) #flatten y ravel aplanan el array pero de manera distinta internamente
print(np.vstack((dimension2, dimension2))) #Concatenar verticalmente, tambien esta hstack
print(np.row_stack((dimension2, dimension1)))
print(np.split(dimension2, 2)) #Dos partes
print(np.all(dimension1 > 3)) #Saber si todos cumplen una condicion, para saber si cualquiera esta np.any
print(np.where(dimension1 > 1)) #Filtrar con todas las ocasiones donde se cumpla la condicion
for i in np.nditer(dimension2): #Iterar array, para poder alterarlo usar op_flags=['readwrite']
    print(i)


