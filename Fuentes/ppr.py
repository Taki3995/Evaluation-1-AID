#----------------------------------------------
# Create Features by use 
# Dispersion Entropy and Permutation entropy
#----------------------------------------------

import pandas   as pd
import numpy   as np
from utility import entropy_dispersion, entropy_prmuta

# Load  parameters Entropy
def conf_entropy():    

    return(opt,d,tau,c)

# Load Data
def load_data(nFile):

# Obtain entropy : dispersión and Permutation
def gets_entropy(x,opt,d,tau,c):
    
    return()
    
# Obtain Features by use Entropy    
def gets_features():
    
    return(F)    
# Beginning ...
def main():
    conf_entropy()            
    load_data()
    F1 = gets_features()
    F2 = gets_features()
    F  = np.concatenate((F1, F2), axis=0)
    save_data(F)
    
       
if __name__ == '__main__':   
	 main()

