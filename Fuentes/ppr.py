#----------------------------------------------
# Create Features by use 
# Dispersion Entropy and Permutation entropy
#----------------------------------------------

import pandas   as pd
import numpy   as np
from utility import entropy_dispersion, entropy_permuta

# Load  parameters Entropy
def conf_entropy():
    config = pd.read_csv("config/conf_ppr.csv", header=None).values.flatten()
    opt = 'dispersion' if config[0] == 1 else 'permutation'
    d = int(config[1])
    tau = int(config[2])
    c = int(config[3])
    W = int(config[4])
    return opt, d, tau, c, W

# Load Data
def load_data(nFile):
    data = pd.read_csv(nFile, header = None)
    return(data)

# Obtain entropy : dispersión and Permutation
def gets_entropy(x, opt, d, tau, c):

    if opt == 'dispersion':
        return entropy_dispersion(x, d, tau, c)
    elif opt == 'permutation':
        return entropy_permuta(x, d, tau)
    else:
        raise ValueError("Opción no válida. ")

# Obtain Features by use Entropy    
def gets_features(file):

    # Se cargan los parametros de configuracion desde conf_entropy
    opt, d, tau, c, W = conf_entropy()

    # Se cargan archivos de datos (class1 o class2)
    data = load_data (f"data/{file}").values.flatten()
    
    # Se recorre en bloques de tamaño W, devolviendo entropia cruzada y normalizada, pero guarda solo la normalizada
    features = []

    for i in range(0, len(data) - W + 1, W):
        segment = data[i:i+W]
        entropies = gets_entropy(segment, opt, d, tau, c)
        features.append(entropies[1])

    return np.array(features).reshape(-1, 1)

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
def main():

    F1 = gets_features("class1.csv")
    F2 = gets_features("class2.csv")
    F  = np.concatenate((F1, F2), axis = 0)
    save_data(F)
    
if __name__ == '__main__':   
	 main()

