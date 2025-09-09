#----------------------------------------------
# Create Features by use 
# Dispersion Entropy and Permutation entropy
#----------------------------------------------

import pandas   as pd
import numpy   as np
from utility import entropy_dispersion, entropy_prmuta

# Load  parameters Entropy
def conf_entropy():
    config = pd.read_csv("config/conf_ppr.csv")["config"].values
    opt = 'dispersion' if config[0] == 1 else 'permutation'
    d = int(config[1])
    tau = int(config[2])
    c = int(config[3])

    return(opt,d,tau,c)

# Load Data
def load_data(nFile):
    data = pd.read_csv(nFile, header = None)
    return(data)

# Obtain entropy : dispersión and Permutation
def gets_entropy(x, opt, d, tau, c):

    if opt == 'dispersion':
        return entropy_dispersion(x, d, tau, c)
    elif opt == 'permutation':
        return entropy_prmuta(x, d, tau)
    else:
        raise ValueError("Opción no válida. ")

# Obtain Features by use Entropy    
def gets_features(file):
    
    return(F)

def save_data(F):
    return

# Beginning ...
def main():
    conf_entropy()            
    load_data()
    F1 = gets_features("class1.csv")
    F2 = gets_features("class2.csv")
    F  = np.concatenate((F1, F2), axis = 0)
    save_data(F)
    
if __name__ == '__main__':   
	 main()

