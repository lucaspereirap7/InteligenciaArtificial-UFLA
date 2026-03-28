import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import time

def rodar_knn_biblioteca(k_valores):
    df = pd.read_csv('iris.csv')
    if 'Id' in df.columns: df = df.drop(columns=['Id'])

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Split 80/20 com semente fixa pra garantir reprodutibilidade
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lista_metricas = []

    for k in k_valores:
        # Treina e prediz para o valor de k atual
        modelo = KNeighborsClassifier(n_neighbors=k)
        modelo.fit(X_train, y_train)
        previsoes = modelo.predict(X_test)

        # Salva as predições em txt pra comparar depois
        pd.DataFrame({'Real': y_test, 'Previsto': previsoes}).to_csv(f"predicoes_sklearn_k{k}.txt", index=False, sep='\t')

        # Gera e salva a matriz de confusão
        cm = confusion_matrix(y_test, previsoes)
        plt.figure(figsize=(7, 5))
        sns.heatmap(cm, annot=True, cmap="OrRd", fmt='d')
        plt.title(f"Matriz Sklearn - K={k}")
        plt.savefig(f"matriz_sklearn_k{k}.png")
        plt.close()

        acc = accuracy_score(y_test, previsoes)
        lista_metricas.append({'k': k, 'acuracia': acc})

    return lista_metricas

if __name__ == "__main__":
    rodar_knn_biblioteca([1, 3, 5, 7])