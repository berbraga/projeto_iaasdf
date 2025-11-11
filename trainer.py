"""
Módulo contendo a função de treinamento da rede neural.
"""
import time
import torch
from torch.utils.data import DataLoader


def treinar_rede(cnn, dataset, epochs=10, learning_rate=0.000001, batch_size=64):
    """
    Treina a rede neural convolucional.
    
    Args:
        cnn: Modelo da rede neural
        dataset: Dataset de treinamento
        epochs: Número de épocas de treinamento
        learning_rate: Taxa de aprendizado
        batch_size: Tamanho do lote
        
    Returns:
        Modelo treinado
    """
    otimizador = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        perda_total = 0
        inicio_tempo = time.time()
        otimizador.zero_grad()
        
        for inputs, targets in train_loader:
            x_hat = cnn(inputs)
            loss = ((targets - x_hat) ** 2).sum()
            
            loss.backward(retain_graph=True)
            perda_total += loss
            
            otimizador.step()
            otimizador.zero_grad()
        
        fim_tempo = time.time()
        perda_media = perda_total / len(dataset)
        tempo_epoch = fim_tempo - inicio_tempo
        print(f"Época {epoch}: Perda Total: {perda_media:.4f}, Tempo: {tempo_epoch:.2f}s")
    
    return cnn

