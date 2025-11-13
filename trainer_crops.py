"""
Módulo contendo a função de treinamento da rede neural para classificação de culturas.
"""
import os
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
    caminho_melhor_modelo = 'melhor_modelo_culturas.pth'
    
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
            try:
                # Tentar salvar usando nome temporário primeiro
                caminho_temp = caminho_melhor_modelo + '.tmp'
                torch.save(modelo.state_dict(), caminho_temp)
                
                # Se sucesso, substituir arquivo antigo
                if os.path.exists(caminho_melhor_modelo):
                    os.remove(caminho_melhor_modelo)
                os.rename(caminho_temp, caminho_melhor_modelo)
                
            except Exception as e:
                print(f"⚠️  Aviso: Não foi possível salvar o melhor modelo: {e}")
                print(f"   Tentando salvar com nome alternativo...")
                try:
                    # Tentar com nome alternativo
                    caminho_alternativo = f'melhor_modelo_culturas_ep{epoch+1}.pth'
                    torch.save(modelo.state_dict(), caminho_alternativo)
                    caminho_melhor_modelo = caminho_alternativo
                    print(f"   ✓ Modelo salvo em '{caminho_alternativo}'")
                except Exception as e2:
                    print(f"   ❌ Erro ao salvar modelo alternativo: {e2}")
        
        print(f"Época {epoch+1}/{epochs}:")
        print(f"  Treino - Loss: {perda_media_treino:.4f}, Acc: {acc_treino:.2f}%")
        print(f"  Validação - Loss: {perda_media_validacao:.4f}, Acc: {acc_validacao:.2f}%")
        print(f"  Tempo: {tempo_epoch:.2f}s")
        print()
    
    # Carregar melhor modelo
    if os.path.exists(caminho_melhor_modelo):
        try:
            modelo.load_state_dict(torch.load(caminho_melhor_modelo, map_location=device))
            print(f"✓ Melhor modelo carregado de '{caminho_melhor_modelo}'")
        except Exception as e:
            print(f"⚠️  Aviso: Não foi possível carregar o melhor modelo: {e}")
            print("   Usando modelo da última época...")
    else:
        print("⚠️  Aviso: Arquivo do melhor modelo não encontrado. Usando modelo da última época...")
    
    print(f"Melhor acurácia de validação: {melhor_acc_validacao:.2f}%")
    
    # Adicionar melhor acurácia ao histórico
    historico['melhor_acc_validacao'] = melhor_acc_validacao
    
    return modelo, historico

