# Logistic Regression's Training :

import numpy as np
import utility as ut
import pandas as pd

#Save weights and Cost
def save_w_cost(W, Cost, fW, fC):
    pd.DataFrame(W).to_csv(fW, index=False, header=False) #guarda pesos en fw (pesos.csv)
    pd.DataFrame(Cost).to_csv(fC, index=False, header=False) #guarda costos en fc (costos.csv) (cada fila representa el costo en una iteracion)
    return

def iniWs(dim):
    W = np.random.randn(dim) #pesos aleatorios distribución normal
    V = np.zeros(dim) #vector momentum en ceros, mismo tamaño que w
    return(W,V)


#Training by use mGD
def train(): 
    # Cargar configuracion
    n_iter, mu, p_train = conf_train()

    # Cargar datos
    X, y = load_data()
    N = len(X)
    L = round(N * p_train)

    # Reordenar de forma aleatoria
    idx = np.random.permutation(N)
    X, y = X[idx], y[idx]

    # Separar en datos de Training y datos de Testing
    Xtrn = X[:L]
    ytrn = y[:L]

    # Guardar archivos
    pd.DataFrame(Xtrn).to_csv("dtrn.csv", index=False, header=False)
    pd.DataFrame(ytrn).to_csv("dtrn_label.csv", index=False, header=False)
    pd.DataFrame(X[L:]).to_csv("dtst.csv", index=False, header=False)
    pd.DataFrame(y[L:]).to_csv("dtst_label.csv", index=False, header=False)

    # Inicializar pesos y momentum
    W, V = iniWs(X.shape[1])
    Cost = []

    # Entrenamiento
    for i in range(n_iter):
        z = 1 / (1 + np.exp(-np.dot(Xtrn, W))) # Regresion Logistica
        error = z - ytrn
        grad = np.dot(Xtrn.T, error) / len(Xtrn)
        V = 0.9 * V - mu * grad
        W += V
        cost = -np.mean(ytrn * np.log(z + 1e-8) + (1 - ytrn) * np.log(1 - z + 1e-8))
        Cost.append(cost)

    return W, Cost


# Load data to train 
def load_data():
    # Carga caracteristicas (x) y etiquetas (y) generadas en pre procesamiento
    X = pd.read_csv("dfeatures.csv", header=None).values
    y = pd.read_csv("label.csv", header=None).values.flatten()
    return (X, y)


# Load config train for Regression
def conf_train():
    #leer conf_train.csv
    config = pd.read_csv("config/conf_train.csv", header=None).values.flatten()

    #Extrae
    n_iter = int(config[0]) # numero de iteraciones
    mu = float(config[1]) # tasa aprendizaje
    p_train = float(config[2]) # proporcion de datos de entrenamiento
    return n_iter, mu, p_train

# Beginning ...
def main():    
    conf_train()
    load_data()
    W, Cost = train()
    save_w_cost(W, Cost, 'pesos.csv','costo.csv')
       
if __name__ == '__main__':   
	 main()

