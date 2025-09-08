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
    ...
    return(W)
# 
def load_data(nFile):
    ....
    return(x,y)
# Beginning ...
def main():			
	load_data()
	load_w()
	zv     = forward(xv,W)      		
	cm,Fsc = metricas(yv,zv) 	
	save_measure(cm,Fsc,'cmatrix.csv','Fscores.csv')		

if __name__ == '__main__':   
	 main()

