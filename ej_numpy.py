import sys
import numpy as np #Base optimizada del ecosistema de datos en Python
np.set_printoptions(threshold = sys.maxsize)

dimension1 = np.array([0,1,2,3]) #Vector
dimension2 = np.array([[0,1,2,3], [4,5,6,7]]) #Matriz
dimension3 = np.array([[[0,1,2,3], [4,5,6,7]], [[8,9,10,11], [12,13,14,15]]]) #Cubo
print(dimension3.ndim) #Cantidad de dimensiones
print(dimension2.shape) #Tamagno en cada eje
print(dimension2.dtype) #Tipo de dato
print(dimension1) #Datos talcual

#eje = np.array([[3,"fg",6], [7]]) #Debe tener el mismo tamagno pero puede tener tipos de datos distintos
print(np.array([3,"g",6, False]).dtype) #Igualmente no es recomendable, siempre es mejor arrays homogeneos
print(np.zeros((2,3), dtype=np.int32)) #Matriz 2x3 lleno de 0s, tambien hay np.ones y np.empty
print(np.eye(5)) #0 en todos menos en la diagonal que son 1
print(np.arange(0, 10, 2)) #Del 0 al 10 en saltos de 2, np.arrange(10) generaria los numeros del 0 al 10
print(np.arange(16).reshape(4,4)) #Matriz generada rapidamente
print(np.ones_like(dimension2)) #Un array con la misma shape pero todo 1s, tambien esta zeros_like
print(np.linspace(0, 1, 5)) #Del 0 al 1, 5 numeros (float)
print(np.random.hypergeometric(ngood=5,nbad=5,nsample=6,size=100)) #Valores aleatorios
copia = dimension2.view() #Crear una copia que en realidad no es igual, solo tiene los mismos valores (pero esta enlazado al original, alteraciones al orignal afectan al view pero cada view es independiente)
copia2 = dimension2.copy() #Crear una copia totalmente desconectada
print(dimension1[[0,1,3,1]]) #Filtrar datos con una mascara
print(dimension3[~np.isnan(dimension3)]) #No incluye los que sean NaN
print(np.count_nonzero(dimension2)) #Cantidad de valores que no son 0
print(np.sort(dimension1)) #Array ordenado

print(dimension3[1,0,2]) #Elemento concreto
print(dimension2[1, :]) #Fila 1 (en general funciona igual que un array en Python)
print(dimension1 * 2) #Calculos a todos los elementos
print(dimension2[:, 0] ** 2)
print(dimension1 + dimension1) #Itera y opera cada elemento (* + - / % ** > < ==), el shape de ambos arrays debe de ser compatible (no necesariamente igual)
print(dimension1.dot(dimension1)) #Dot product
print(np.sin(dimension2)) #Seno a cada valor, tambien estan exp, sqrt, arcsin, cos y tan

print(np.sum(dimension2)) #Suma de todos los elementos
print(np.average(dimension3)) #La media total (similar a np.mean, no confundir con np.median que coje el del medio), hay mas funciones de agregacion (como var o std)
print(np.sum(dimension2, axis=0)) #Suma hacia un eje concreto, de esta manera no devolvera un solo valor (0 columnas, 1 filas, etc)
print(np.std(dimension3)) #Desviacion estandar global
print(np.argmax(dimension2)) #Numero maximo, tambien esta argmin, o np.max y np.min
print(np.cumsum(dimension2, axis=1)) #Suma acumulativa
print(np.percentile(dimension1, 50)) #Calculo de percentil
print(np.all(dimension1 > 2)) #True si todos cumplen la condicion, any es para cualquiera
print(np.sum(np.array([True, False, True, False]))) #En la mayoria de operaciones, True es 1 y False es 0

filtro = dimension1 > 2 #True para los que cumplan la condicion, devuelve un array de las mismas propiedades, se puede usar como mascara
print(filtro) 
print(dimension1[filtro]) #Solo las que respectivamente sean true
print(np.where(dimension1 > 1, "mas", "menos")) #Valor custom en lugar de booleano
nuevo = dimension2.reshape(1, 8) #Redimensionar array, el contenido es igual, debe terminar teniendo el mismo numero de elementos y se pueden poner o quitar dimensiones, si una es -1 lo inferira segun sea necesario
print(dimension2.T) #Girar array
print(dimension2.flatten()) #flatten y ravel aplanan el array pero de manera distinta internamente
print(np.vstack((dimension2, dimension2))) #Concatenar verticalmente, tambien esta hstack
print(np.concatenate((dimension1, dimension1))) #Concatenar directamente
print(np.row_stack((dimension2, dimension1)))
print(np.split(dimension2, 2)) #Dividir en partes, se puede usar una tupla para especificar puntos de corte concretos, tambien estan hsplit y vsplit para dimensiones concretas
print(np.all(dimension1 > 3)) #Saber si todos cumplen una condicion, para saber si cualquiera esta np.any
print(np.where(dimension1 > 1)) #Filtrar con todas las ocasiones donde se cumpla la condicion
for i in np.nditer(dimension2): #Iterar array, para poder alterarlo usar op_flags=['readwrite']
    print(i)

import matplotlib.pyplot as plt
imagen = plt.imread('./imagen.png') #Pixeles de imagen en array de Numpy
print(imagen.shape) #x, y, rgba
imagen[:,:,0] = imagen[:,:,0] * 2 #Transformaciones tipo shader
plt.imshow(imagen)
plt.show()
