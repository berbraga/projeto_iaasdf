"""
Módulo para avaliar o modelo treinado.
"""
import torch
from torch.utils.data import DataLoader


def avaliar_modelo(cnn, dataset, threshold=0.5):
    """
    Avalia o modelo e gera uma matriz de confusão.
    
    Args:
        cnn: Modelo treinado
        dataset: Dataset para avaliação
        threshold: Limiar para classificação (padrão: 0.5)
        
    Returns:
        dict: Dicionário com a matriz de confusão e métricas
    """
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Matriz de confusão: [predito][real]
    # [0][0] = verdadeiro negativo, [0][1] = falso negativo
    # [1][0] = falso positivo, [1][1] = verdadeiro positivo
    matriz_confusao = [[0, 0], [0, 0]]
    
    for inputs, targets in train_loader:
        x_hat = cnn(inputs)
        classe_predita = 0
        if x_hat[0] > threshold:
            classe_predita = 1
        
        classe_real = int(targets[0][0])
        matriz_confusao[classe_predita][classe_real] += 1
    
    verdadeiros_positivos = matriz_confusao[1][1]
    verdadeiros_negativos = matriz_confusao[0][0]
    falsos_positivos = matriz_confusao[1][0]
    falsos_negativos = matriz_confusao[0][1]
    
    total = verdadeiros_positivos + verdadeiros_negativos + falsos_positivos + falsos_negativos
    acuracia = (verdadeiros_positivos + verdadeiros_negativos) / total if total > 0 else 0
    precisao = verdadeiros_positivos / (verdadeiros_positivos + falsos_positivos) if (verdadeiros_positivos + falsos_positivos) > 0 else 0
    recall = verdadeiros_positivos / (verdadeiros_positivos + falsos_negativos) if (verdadeiros_positivos + falsos_negativos) > 0 else 0
    
    resultados = {
        'matriz_confusao': matriz_confusao,
        'verdadeiros_positivos': verdadeiros_positivos,
        'verdadeiros_negativos': verdadeiros_negativos,
        'falsos_positivos': falsos_positivos,
        'falsos_negativos': falsos_negativos,
        'acuracia': acuracia,
        'precisao': precisao,
        'recall': recall
    }
    
    return resultados


def imprimir_resultados(resultados):
    """
    Imprime os resultados da avaliação de forma formatada.
    
    Args:
        resultados: Dicionário com os resultados da avaliação
    """
    print("\n" + "="*50)
    print("RESULTADOS DA AVALIAÇÃO")
    print("="*50)
    print(f"\nMatriz de Confusão:")
    print(f"                Real: Não-Pássaro  Pássaro")
    print(f"Predito: Não-Pássaro      {resultados['matriz_confusao'][0][0]:6d}      {resultados['matriz_confusao'][0][1]:6d}")
    print(f"Predito: Pássaro          {resultados['matriz_confusao'][1][0]:6d}      {resultados['matriz_confusao'][1][1]:6d}")
    print(f"\nMétricas:")
    print(f"  Verdadeiros Positivos: {resultados['verdadeiros_positivos']}")
    print(f"  Verdadeiros Negativos: {resultados['verdadeiros_negativos']}")
    print(f"  Falsos Positivos: {resultados['falsos_positivos']}")
    print(f"  Falsos Negativos: {resultados['falsos_negativos']}")
    print(f"  Acurácia: {resultados['acuracia']:.4f}")
    print(f"  Precisão: {resultados['precisao']:.4f}")
    print(f"  Recall: {resultados['recall']:.4f}")
    print("="*50 + "\n")

