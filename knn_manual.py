import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Cálculo da distância euclidiana entre dois pontos
def calcular_distancia(ponto_a, ponto_b):
    soma_quadrados = 0
    for i in range(len(ponto_a)):
        soma_quadrados += (ponto_a[i] - ponto_b[i]) ** 2
    return soma_quadrados ** 0.5

def meu_knn_manual(k_lista):
    dados = pd.read_csv('iris.csv')
    if 'Id' in dados.columns:
        dados = dados.drop(columns=['Id'])

    # Embaralhando os dados
    dados_random = dados.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split de 80% para treino e 20% para teste
    corte = int(len(dados_random) * 0.8)
    treino = dados_random.iloc[:corte]
    teste = dados_random.iloc[corte:]

    X_treino = treino.iloc[:, :-1].values
    y_treino = treino.iloc[:, -1].values
    X_teste = teste.iloc[:, :-1].values
    y_teste = teste.iloc[:, -1].values

    resultados_finais = []

    for k in k_lista:
        print(f"Rodando para K = {k}...")
        predicoes = []

        for p_teste in X_teste:
            distancias = []
            for i in range(len(X_treino)):
                d = calcular_distancia(p_teste, X_treino[i])
                distancias.append((d, y_treino[i]))
            
            # Ordena por distância e pega os K vizinhos
            distancias.sort(key=lambda x: x[0])
            vizinhos = [v[1] for v in distancias[:k]]
            
            # Votação da classe majoritária
            voto = max(set(vizinhos), key=vizinhos.count)
            predicoes.append(voto)

        # Exportando predições para txt
        saida_txt = pd.DataFrame({'Real': y_teste, 'Previsto': predicoes})
        saida_txt.to_csv(f"predicoes_manual_k{k}.txt", index=False, sep='\t')

        # Gerando a matriz de confusão com crosstab
        matriz = pd.crosstab(y_teste, np.array(predicoes), rownames=['Real'], colnames=['Previsto'])
        plt.figure(figsize=(7, 5))
        sns.heatmap(matriz, annot=True, cmap="YlGnBu", fmt='d')
        plt.title(f"Matriz de Confusão Manual - K={k}")
        plt.savefig(f"matriz_manual_k{k}.png")
        plt.close()

        acuracia = np.mean(y_teste == np.array(predicoes))
        resultados_finais.append({'k': k, 'acuracia': acuracia, 'preds': predicoes})

    return resultados_finais, y_teste

if __name__ == "__main__":
    inicio = time.time()
    meu_knn_manual([1, 3, 5, 7])
    print(f"Tempo total: {time.time() - inicio:.2f}s")