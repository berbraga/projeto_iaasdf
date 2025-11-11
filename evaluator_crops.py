"""
Módulo para avaliar o modelo treinado em classificação de culturas.
"""
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


def avaliar_modelo(modelo, dataset, classes, device='cpu', batch_size=32):
    """
    Avalia o modelo e gera métricas detalhadas.
    
    Args:
        modelo: Modelo treinado
        dataset: Dataset para avaliação
        classes: Lista com nomes das classes
        device: Dispositivo ('cpu' ou 'cuda')
        batch_size: Tamanho do lote
        
    Returns:
        dict: Dicionário com métricas e matriz de confusão
    """
    modelo.eval()
    modelo = modelo.to(device)
    
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    todas_predicoes = []
    todos_labels = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = modelo(inputs)
            _, preditos = torch.max(outputs, 1)
            
            todas_predicoes.extend(preditos.cpu().numpy())
            todos_labels.extend(targets.cpu().numpy())
    
    # Calcular métricas
    todas_predicoes = np.array(todas_predicoes)
    todos_labels = np.array(todos_labels)
    
    acuracia = 100 * np.sum(todas_predicoes == todos_labels) / len(todos_labels)
    
    # Matriz de confusão
    matriz_confusao = confusion_matrix(todos_labels, todas_predicoes)
    
    # Relatório de classificação
    relatorio = classification_report(
        todos_labels, 
        todas_predicoes, 
        target_names=classes,
        output_dict=True
    )
    
    resultados = {
        'acuracia': acuracia,
        'matriz_confusao': matriz_confusao,
        'relatorio': relatorio,
        'predicoes': todas_predicoes,
        'labels': todos_labels
    }
    
    return resultados


def imprimir_resultados(resultados, classes):
    """
    Imprime os resultados da avaliação de forma formatada.
    
    Args:
        resultados: Dicionário com os resultados da avaliação
        classes: Lista com nomes das classes
    """
    print("\n" + "="*70)
    print("RESULTADOS DA AVALIAÇÃO")
    print("="*70)
    print(f"\nAcurácia Geral: {resultados['acuracia']:.2f}%")
    
    print("\n" + "-"*70)
    print("RELATÓRIO POR CLASSE:")
    print("-"*70)
    
    relatorio = resultados['relatorio']
    
    # Imprimir métricas por classe
    print(f"{'Classe':<25} {'Precisão':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*70)
    
    for classe in classes:
        if classe in relatorio:
            prec = relatorio[classe]['precision'] * 100
            rec = relatorio[classe]['recall'] * 100
            f1 = relatorio[classe]['f1-score'] * 100
            print(f"{classe:<25} {prec:>10.2f}% {rec:>10.2f}% {f1:>10.2f}%")
    
    print("-"*70)
    print(f"{'MÉDIA':<25} {relatorio['macro avg']['precision']*100:>10.2f}% "
          f"{relatorio['macro avg']['recall']*100:>10.2f}% "
          f"{relatorio['macro avg']['f1-score']*100:>10.2f}%")
    print("="*70 + "\n")

