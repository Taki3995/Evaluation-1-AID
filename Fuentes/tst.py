# Testing for Logistic Regresion
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
    W = np.loadtxt("pesos.csv", delimiter = ',')
    return(W)

# Load test 
def load_data(nFile):
    data = np.loadtxt(nFile, delimiter = ',')
    xv = np.loadtxt("dtst.csv", delimiter=',') # xv e yv ya que cada archivo de datos de entrenamiento y testing esta separado en caracteristicas y etiquetas
    yv = np.loadtxt("dtst_label.csv", delimiter=',') 
    return(xv, yv)

# Beginning ...
def main():			
	xv, yv = load_data()
	W = load_w()
	zv     = forward(xv,W)      		
	cm,Fsc = metricas(yv,zv) 	
	save_measure(cm,Fsc,'cmatrix.csv','Fscores.csv')		

if _name_ == '_main_':   
	 main()