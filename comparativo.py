import time
from knn_manual import meu_knn_manual
from knn_sklearn import rodar_knn_biblioteca

def realizar_comparativo():
    ks = [1, 3, 5, 7]

    # Roda o manual e mede o tempo
    print("Executando teste manual...")
    t_inicio_hc = time.time()
    res_hc, y_real = meu_knn_manual(ks)
    t_fim_hc = time.time() - t_inicio_hc

    # Roda o sklearn e mede o tempo
    print("\nExecutando teste com sklearn...")
    t_inicio_sk = time.time()
    res_sk = rodar_knn_biblioteca(ks)
    t_fim_sk = time.time() - t_inicio_sk

    # Exibe o resumo da comparação
    print("\n" + "="*30)
    print("RESULTADOS FINAIS")
    print("="*30)
    print(f"Tempo Manual: {t_fim_hc:.4f}s")
    print(f"Tempo Sklearn: {t_fim_sk:.4f}s")
    
    # Acurácia por valor de K
    for i in range(len(ks)):
        print(f"\nK = {ks[i]}:")
        print(f"  Acurácia Manual: {res_hc[i]['acuracia']:.4f}")
        print(f"  Acurácia Sklearn: {res_sk[i]['acuracia']:.4f}")

if __name__ == "__main__":
    realizar_comparativo()