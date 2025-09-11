# Logistic Regression's Training :

import numpy as np
import utility as ut
import pandas as pd

#Save weights and Cost
def save_w_cost(W, Cost, fW, fC): # W calculado con iniWs() en train(), Cost calculado en train()
    pd.DataFrame(W).to_csv(fW, index=False, header=False) #guarda pesos en fw (pesos.csv)
    pd.DataFrame(Cost).to_csv(fC, index=False, header=False) #guarda costos en fc (costos.csv) (cada fila representa el costo en una iteracion)
    return

def iniWs(dim):
    W = np.random.randn(dim) # Se generan pesos aleatorios distribución normal
    V = np.zeros(dim) # Vector momentum en ceros, mismo tamaño que w
    return(W,V)

#Training by use mGD
def train(X, y, n_iter, mu, p_train): # X es la matriz de caracteristicas, N muestras x D caracteriticas
    # Normalizar features para evitar gradientes diminutos
    X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
    
    # Cargar datos
    N = len(X) # Total muestras
    L = round(N * p_train) # Calcula cuantas muestras usar para training, el resto se usan en testing

    # Mezclar de forma aleatoria los dtos para evitar sesgos
    idx = np.random.permutation(N)
    X, y = X[idx], y[idx]

    # Separar en datos de Training y datos de Testing
    Xtrn = X[:L]
    ytrn = y[:L]

    Xtst = X[L:]
    ytst = y[L:]

    Xtrn_aug = np.hstack([Xtrn, np.ones((Xtrn.shape[0], 1))])
    Xtst_aug = np.hstack([Xtst, np.ones((Xtst.shape[0], 1))])

    # Guardar datos de training y testing
    pd.DataFrame(Xtrn_aug).to_csv("dtrn.csv", index=False, header=False)
    pd.DataFrame(ytrn).to_csv("dtrn_label.csv", index=False, header=False)
    pd.DataFrame(Xtst_aug).to_csv("dtst.csv", index=False, header=False)
    pd.DataFrame(ytst).to_csv("dtst_label.csv", index=False, header=False)
 
    # Inicializar pesos y momentum
    W, V = iniWs(Xtrn_aug.shape[1])
    Cost = []
    eps = 1e-8

    # Entrenamiento
    for i in range(n_iter):
        z = 1 / (1 + np.exp(-np.dot(Xtrn_aug, W))) # Regresion Logistica (sigmoide, combinacion lineal en probabilidades)
        error = z - ytrn # Error entre prediccion y etiqueta
        grad = np.dot(Xtrn_aug.T, error) / len(Xtrn_aug) # Gradiente del costo respecto a pesos
        V = 0.9 * V - mu * grad # Acumula gradiente suavizado
        W += V # Actualiza vector momentum
        cost = -np.mean(ytrn * np.log(z + eps) + (1 - ytrn) * np.log(1 - z + eps)) # Calcula cross entropy
        #cost = -np.mean(ytrn * np.log(z) + (1 - ytrn) * np.log(1 - z))
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
    n_iter, mu, p_train = conf_train()
    X, y = load_data()
    W, Cost = train(X, y, n_iter, mu, p_train)
    save_w_cost(W, Cost, 'pesos.csv','costo.csv')
       
if __name__ == '__main__':   
	 main()

