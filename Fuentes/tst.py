# Testing for Logistic Regresion
import numpy as np
from plotst import plot_cost, plot_confusion, plot_metrics

def forward(xv,w): # Aplica Regresion Logistica
    zv = 1 / (1 + np.exp(-np.dot(xv, w)))
    return(zv)

def measure(yv,zv):
    y_pred = (zv >= 0.5).astype(int)

    cmatrix = np.zeros((2,2), dtype = int) 

    for yt, yp in zip(yv, y_pred): # Recorrer tupla de valor usando zip (true label vs predicted)
        cmatrix[int(yt), int(yp)] += 1 # Matriz de confusion
    
    for i in [0, 1]: # Busca valor en matriz de confusion
        TP = cmatrix[i, i]
        FP = cmatrix[1 - i, i]
        FN = cmatrix[i, 1 - i]

    Fscores = []
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    F = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    Fscores.append(F)

    return(cmatrix, np.array(Fscores))

def save_measure(cm,Fsc,nFile1,nFile2):
    np.savetxt(nFile1, cm, fmt="%d", delimiter=",")
    np.savetxt(nFile2, Fsc.reshape(1, -1), fmt="%.4f", delimiter=",")

    return()

# Load weight
def load_w():
    W = np.loadtxt("pesos.csv", delimiter = ',')
    return(W)

# Load test 
def load_data():
    #data = np.loadtxt(nFile, delimiter = ',')
    xv = np.loadtxt("dtst.csv", delimiter=',') # xv e yv ya que cada archivo de datos de entrenamiento y testing esta separado en caracteristicas y etiquetas
    yv = np.loadtxt("dtst_label.csv", delimiter=',') 
    return(xv, yv)



# Beginning ...
def main():
    xv, yv = load_data()
    W = load_w()
    zv = forward(xv, W)
    cm, Fsc = measure(yv, zv)
    save_measure(cm, Fsc, 'cmatrix.csv', 'Fscores.csv')

    # Visualización
    plot_cost()
    plot_confusion()

if __name__ == '__main__':   
	main()
    