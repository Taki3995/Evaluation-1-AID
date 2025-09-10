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