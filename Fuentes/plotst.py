import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Evolución del costo
def plot_cost():
    cost = pd.read_csv("costo.csv", header=None).values.flatten()
    plt.figure(figsize=(8, 4))
    plt.plot(cost, marker='o', linestyle='-', color='blue')
    plt.title("Evolución del Costo durante el Entrenamiento")
    plt.xlabel("Iteración")
    plt.ylabel("Costo (Cross Entropy)")
    plt.grid(True)
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

# 3. Métricas de desempeño
def plot_metrics():
    metrics = pd.read_csv("Fscores.csv", header=None).values.flatten()
    labels = ["Precisión", "Recall", "F-score"]
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, metrics, color=["green", "orange", "purple"])
    plt.ylim(0, 1.05)
    plt.title("Métricas de Desempeño")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.4f}", ha='center')
    plt.tight_layout()
    plt.savefig("grafico_metricas.png")
    plt.show()