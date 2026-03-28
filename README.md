# Trabalho Prático 01 - Classificação KNN (Inteligência Artificial)

Este projeto implementa o algoritmo K-Nearest Neighbors (KNN) para a classificação do conjunto de dados Iris. O objetivo é comparar uma implementação manual ("hardcore") com a implementação da biblioteca Scikit-Learn, avaliando métricas de desempenho e tempo de execução.

## Estrutura de Arquivos

*   **knn_manual.py**: Implementação do algoritmo KNN do zero, utilizando apenas NumPy e Pandas para manipulação de dados. Realiza o cálculo da distância euclidiana e a votação de classes.
*   **knn_sklearn.py**: Implementação utilizando a biblioteca Scikit-Learn para validação dos resultados obtidos na versão manual.
*   **comparativo.py**: Script que integra as duas versões, executa os testes para K = {1, 3, 5, 7} e compara o tempo de processamento de cada abordagem.
*   **iris.csv**: Base de dados Iris utilizada para o treinamento e teste (divisão 80/20).

## Requisitos e Instalação

Certifique-se de ter o Python instalado. As bibliotecas necessárias podem ser instaladas via terminal com o comando abaixo:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Como Executar

Para rodar o projeto completo e gerar as comparações, execute o arquivo principal:

```bash
python comparativo.py
```

## Resultados Gerados

Após a execução, o projeto gera automaticamente na pasta raiz:
1.  **Imagens (.png)**: Matrizes de confusão para cada valor de K, tanto para a versão manual quanto para a versão Sklearn.
2.  **Relatórios (.txt)**: Arquivos contendo as predições reais vs. previstas para cada configuração de K.
3.  **Console**: Exibição das acurácias e do tempo de execução de cada classificador.

## Análise Resumida

*   A implementação manual atingiu resultados de acurácia equivalentes aos da biblioteca oficial, validando a lógica de cálculo de distâncias.
*   O Scikit-Learn apresentou maior eficiência em tempo de execução devido às otimizações internas da biblioteca.
*   As matrizes de confusão demonstram que o modelo classifica com alta precisão as três espécies de Iris, com pequenas variações apenas nas zonas de fronteira entre Versicolor e Virginica.
