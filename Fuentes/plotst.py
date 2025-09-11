import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Evolución del costo

def plot_cost():
    cost = pd.read_csv("costo.csv", header=None).values.flatten()

    plt.figure(figsize=(8, 4))
    plt.plot(range(len(cost)), cost, color='black', linewidth=0.8)  # Línea más fina
    plt.title("Training: Reg.Logistic", fontsize=14)
    plt.xlabel("Iterations", fontsize=12)
    plt.ylabel("Binary Entropy", fontsize=12)
    plt.xlim(0, 5000)
    plt.ylim(0, 1.2)
    plt.xticks([0, 1000, 2000, 3000, 4000, 5000])
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    plt.tight_layout()
    plt.savefig("grafico_costo.png")
    plt.show()


# 2. Matriz de Confusión
def plot_confusion():
    cm = pd.read_csv("cmatrix.csv", header=None).values
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Pred: 0", "Pred: 1"],
                yticklabels=["Real: 0", "Real: 1"])
    plt.title("Matriz de Confusión")
    plt.tight_layout()
    plt.savefig("grafico_confusion.png")
    plt.show()