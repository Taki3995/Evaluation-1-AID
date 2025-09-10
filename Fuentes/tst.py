[7:17 a.m., 3/9/2025] Aleeee💩: Análisis Crítico (Casos Internacionales)

Si miramos otros países, como Estados Unidos con FirstNet o Europa con T-Priority, vemos que ya tienen LTE prioritario para fuerzas de seguridad.

La clave no ha sido inventar nuevas tecnologías, sino coordinar políticas, interoperar con múltiples proveedores y garantizar prioridad de tráfico en emergencias.
[7:21 a.m., 3/9/2025] Aleeee💩: Conclusiones

La infraestructura policial tiene cimientos sólidos, pero todavía no es suficiente.

La combinacion del cifrado fuerte y redundancia integral es la mejor vía para asegurar continuidad y resiliencia.
Lo positivo es que muchas de las mejoras son factibles sin necesidad de un rediseño total. Basta con fortalecer lo que ya tenemos y adoptar estándares internacionales.
[11:13 a.m., 3/9/2025] Aleeee💩: Joel#2017, mercado pago
[4:34 p.m., 5/9/2025] Aleeee💩: Allowance = mezada
[11:05 p.m., 7/9/2025] Aleeee💩: def entropy_dispersion(x,d,tau,c):
    ...
    return(entr)
#


# Permutation Entropy
def entropy_permuta(x, m, tau):

    # Parametros
    x = np.asarray(x)
    n = len(x)
    if n < (m - 1) * tau + 1:
        raise ValueError("La serie es demasiado corta para los parámetros dados.")

    # Paso 1: Crear la matriz de permutaciones
    patterns = []

    for i in range(n - (m - 1) * tau):
        window = x[i : i + tau * m : tau] # 2a Crear vector-embedding
        #ranks = pd.Series(window).rank(method='first').astype(int).values # 2b
        #pattern = tuple(ranks - 1)
        pattern = tuple(np.argsort(window)) #2b Ordena los elementos
        patterns.append(pattern)
    
    # Paso 3: Contar la frecuencia de cada patrón
    df = pd.Series(patterns)
    …
[11:42 p.m., 7/9/2025] Aleeee💩: https://git-scm.com/downloads/win
[12:44 a.m., 8/9/2025] Aleeee💩: # My Utility : auxiliars functions

import pandas as pd
import numpy  as np

# Cresidual-Dispersion Entropy
def entropy_dispersion(x,d,tau,c):
    # Parametros
    x = np.asarray(x)
    n = len(x)

    if n < (d - 1) * tau + 1:
        raise ValueError("La serie es demasiado corta para los parámetros dados.")
    
    # Paso 1: Normalizar el vector
    x_norm = (x - np.min(x)) / (np.max(x) - np.min(x))

    # Paso 2: Crear vectores-embedding
    embeddings = []
    for i in range(n - (d - 1) * tau):
        window = x_norm[i : i + tau * d : tau]
        embeddings.append(window)

    # Paso 3: Mapear cada vector-embedding
    y = [np.round(c * emb + 0.5).astype(int) for emb in embeddings]

    # Paso 4: Convertir el vector Y en un número
    pattern = []

  …
[1:19 p.m., 8/9/2025] Aleeee💩: Si nos da 100
[1:19 p.m., 8/9/2025] Aleeee💩: Esta buena
[10:56 p.m., 9/9/2025] Aleeee💩: # Testing for Logistic Regresion
import numpy as np

def forward(xv,w):
    ...
    return(zv)
#
def measure(yv,zv):
    ...
    return(cmatrix,Fscores)
#
def save_measure(cm,Fsc,nFile1,nFile2):
    ...
    return()
# Load weight
def load_w(nFile):
    W = np.loadtxt(nFile, delimiter = ',')
    return(W)

# Load test 
def load_data(nFile):
    data = np.loadtxt(nFile, delimiter = ',')
    x = data[:, :-1] # Todas las columnas menos la ultima
    y = data[:, -1]  # Solo la ultima columna
    return(x, y)

# Beginning ...
def main():			
	load_data()
	load_w()
	zv     = forward(xv,W)      		
	cm,Fsc = metricas(yv,zv) 	
	save_measure(cm,Fsc,'cmatrix.csv','Fscores.csv')		

if _name_ == '_main_':   
	 main()