
# My Utility : auxiliars functions

import pandas as pd
import numpy  as np

# CResidual-Dispersion Entropy
def entropy_dispersion(x,d,tau,c):
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
    freq = df.value_counts(normalize=True)

    # Paso 4: Calcular la entropía de Shannon
    entr = -np.sum(freq * np.log2(freq))

    n_entr = entr / np.log2(np.math.factorial(m))  # 5a Entropía normalización

    return(n_entr)