#----------------------------------------------
# Create Features by use 
# Dispersion Entropy and Permutation entropy
#----------------------------------------------

import pandas   as pd
import numpy   as np
from utility import entropy_dispersion, entropy_permuta

# Carga parametros Entropy
def conf_entropy():
    config = pd.read_csv("config/conf_ppr.csv", header=None).values.flatten()
    opt = 'dispersion' if config[0] == 1 else 'permutation'# Tipo de entropía (0-dispersión/1-permuta).
    d = int(config[1]) # Dimensión embebida
    tau = int(config[2]) #Tiempo de retardo embebido
    c = int(config[3]) # Número de clase de Entropía Dispersión
    W = int(config[4]) # Tamaño de Segmentación de los archivos clases
    return opt, d, tau, c, W

# Carga datos class1 y class2
def load_data():
    data1 = pd.read_csv("data/class1.csv", header=None).values # matriz tamaño (N1, L)
    data2 = pd.read_csv("data/class2.csv", header=None).values # matriz tamaño (N2, L)
    return data1, data2

# Segun la opcion, calcula la entropia llamando a la funcion elegida. 
def gets_entropy(x, opt, d, tau, c):

    if opt == 'dispersion':
        return entropy_dispersion(x, d, tau, c)
    elif opt == 'permutation':
        return entropy_permuta(x, d, tau)
    else:
        raise ValueError("Opción no válida. ")

# Obtain Features by use Entropy    
def gets_features(data, opt, d, tau, c, W):
    N, L = data.shape
    K = N // W # Divide matriz en k bloques de tamaño w filas c/u
    feats = np.zeros((K, W)) # Matriz vacia para guardar caracteristicas
    
    for k in range(K): #extrae bloques de tamaño (W, L)
        block = data[k*W:(k+1)*W, :] 

        # Para cada fila del bloque, se calcula entropia sobre su serie 1D de largo L
        for j in range(W):
            row_series = block[j, :]
            feats[k, j] = gets_entropy(row_series, opt, d, tau, c) # guarda valor en feats
    return feats  # Devuelve matriz de caracteristicas (K, W)

def save_data(F):
    # se divide en caracteristicas clase 1 (etiqueta 1) y clase 2 (etiqueta 0)
    n1 = F.shape[0] // 2
    F1 = F[:n1]
    F2 = F[n1:]

    pd.DataFrame(F1).to_csv("dfeatures1.csv", index=False, header=False)
    pd.DataFrame(F2).to_csv("dfeatures2.csv", index=False, header=False)
    pd.DataFrame(F).to_csv("dfeatures.csv", index=False, header=False) # Caracteristicas concatenadas

    labels = np.array([1]*len(F1) + [0]*len(F2))
    pd.DataFrame(labels).to_csv("label.csv", index=False, header=False)

# Beginning ...
def main(): ##
    opt, d, tau, c, W = conf_entropy()
    data1, data2 = load_data()
    F1 = gets_features(data1, opt, d, tau, c, W)
    F2 = gets_features(data2, opt, d, tau, c, W)
    F  = np.concatenate((F1, F2), axis=0)
    save_data(F)
    
if __name__ == '__main__':   
	 main()

