"""
Módulo contendo a função de treinamento da rede neural para classificação de culturas.
"""
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def treinar_rede(modelo, dataset_treino, dataset_validacao, epochs=50, 
                 learning_rate=0.001, batch_size=32, device='cpu'):
    """
    Treina a rede neural convolucional com validação.
    
    Args:
        modelo: Modelo da rede neural
        dataset_treino: Dataset de treinamento
        dataset_validacao: Dataset de validação
        epochs: Número de épocas de treinamento
        learning_rate: Taxa de aprendizado
        batch_size: Tamanho do lote
        device: Dispositivo ('cpu' ou 'cuda')
        
    Returns:
        Modelo treinado e histórico de métricas
    """
    modelo = modelo.to(device)
    criterio = nn.CrossEntropyLoss()
    otimizador = torch.optim.Adam(modelo.parameters(), lr=learning_rate)
    
    train_loader = DataLoader(dataset_treino, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset_validacao, batch_size=batch_size, shuffle=False)
    
    historico = {
        'treino_loss': [],
        'treino_acc': [],
        'validacao_loss': [],
        'validacao_acc': []
    }
    
    melhor_acc_validacao = 0.0
    
    for epoch in range(epochs):
        # Fase de treinamento
        modelo.train()
        perda_treino = 0.0
        corretos_treino = 0
        total_treino = 0
        
        inicio_tempo = time.time()
        
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            otimizador.zero_grad()
            outputs = modelo(inputs)
            loss = criterio(outputs, targets)
            loss.backward()
            otimizador.step()
            
            perda_treino += loss.item()
            _, preditos = torch.max(outputs.data, 1)
            total_treino += targets.size(0)
            corretos_treino += (preditos == targets).sum().item()
        
        # Fase de validação
        modelo.eval()
        perda_validacao = 0.0
        corretos_validacao = 0
        total_validacao = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = modelo(inputs)
                loss = criterio(outputs, targets)
                
                perda_validacao += loss.item()
                _, preditos = torch.max(outputs.data, 1)
                total_validacao += targets.size(0)
                corretos_validacao += (preditos == targets).sum().item()
        
        # Calcular métricas
        acc_treino = 100 * corretos_treino / total_treino
        acc_validacao = 100 * corretos_validacao / total_validacao
        perda_media_treino = perda_treino / len(train_loader)
        perda_media_validacao = perda_validacao / len(val_loader)
        
        historico['treino_loss'].append(perda_media_treino)
        historico['treino_acc'].append(acc_treino)
        historico['validacao_loss'].append(perda_media_validacao)
        historico['validacao_acc'].append(acc_validacao)
        
        fim_tempo = time.time()
        tempo_epoch = fim_tempo - inicio_tempo
        
        # Salvar melhor modelo
        if acc_validacao > melhor_acc_validacao:
            melhor_acc_validacao = acc_validacao
            torch.save(modelo.state_dict(), 'melhor_modelo_culturas.pth')
        
        print(f"Época {epoch+1}/{epochs}:")
        print(f"  Treino - Loss: {perda_media_treino:.4f}, Acc: {acc_treino:.2f}%")
        print(f"  Validação - Loss: {perda_media_validacao:.4f}, Acc: {acc_validacao:.2f}%")
        print(f"  Tempo: {tempo_epoch:.2f}s")
        print()
    
    # Carregar melhor modelo
    modelo.load_state_dict(torch.load('melhor_modelo_culturas.pth'))
    print(f"Melhor acurácia de validação: {melhor_acc_validacao:.2f}%")
    
    return modelo, historico

