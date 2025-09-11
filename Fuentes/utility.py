
# My Utility : auxiliars functions

import pandas as pd
import numpy  as np

# CResidual-Dispersion Entropy
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

    for s in y:
        base_k = sum((s[j] - 1) * (c ** (d - j - 1)) for j in range(d))
        pattern.append(base_k)

    # Paso 5: Contar la frecuencia de cada patrón
    df = pd.Series(pattern)

    # Paso 6: Calcular la probabilidad de cada patrón
    p = df.value_counts(normalize = True)

    # Paso 7: Calcular la entropía    
    entr = - np.sum(p * np.log2(p))

    # Paso 8 : Normalización
    r = c ** d
    n_enter = entr / np.log2(r)

    return(n_enter)

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
        pattern = tuple(np.argsort(window)) #2b Ordena los elementos
        patterns.append(pattern)
    
    # Paso 3: Contar la frecuencia de cada patrón
    df = pd.Series(patterns)
    freq = df.value_counts(normalize=True)

    # Paso 4: Calcular la entropía de Shannon
    entr = -np.sum(freq * np.log2(freq))

    n_entr = entr / np.log2(np.math.factorial(m))  # 5a Entropía normalización

    return(n_entr)