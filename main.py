"""
Script principal para executar o treinamento e avaliação do modelo de classificação de pássaros.
"""
import torch
from model import RedeCnnBirdNotBird
from data_loader import preparar_dataset
from trainer import treinar_rede
from evaluator import avaliar_modelo, imprimir_resultados


def main():
    """
    Função principal que executa o pipeline completo:
    1. Detecta o dispositivo (GPU ou CPU)
    2. Carrega e prepara os dados
    3. Treina o modelo
    4. Avalia o modelo
    """
    # Configurações
    zip_path_passaros = 'bird.zip'
    zip_path_nao_passaros = 'not-bird.zip'
    max_imagens_por_classe = 1000
    epochs = 100
    learning_rate = 0.000001
    batch_size = 64
    
    # Detectar dispositivo
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"GPU está {'disponível' if device == 'cuda' else 'NÃO disponível'}")
    print(f"Usando dispositivo: {device}\n")
    
    # Carregar e preparar dados
    print("="*50)
    print("CARREGANDO DADOS")
    print("="*50)
    dataset = preparar_dataset(
        zip_path_passaros,
        zip_path_nao_passaros,
        max_imagens_por_classe,
        device
    )
    
    # Criar modelo
    print("\n" + "="*50)
    print("CRIANDO MODELO")
    print("="*50)
    cnn = RedeCnnBirdNotBird().to(device)
    print("Modelo criado e movido para", device)
    
    # Treinar modelo
    print("\n" + "="*50)
    print("TREINANDO MODELO")
    print("="*50)
    cnn = treinar_rede(
        cnn,
        dataset,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        device=device
    )
    
    # Avaliar modelo
    print("\n" + "="*50)
    print("AVALIANDO MODELO")
    print("="*50)
    resultados = avaliar_modelo(cnn, dataset, device=device)
    imprimir_resultados(resultados)
    
    # Salvar modelo (opcional)
    # torch.save(cnn.state_dict(), 'modelo_treinado.pth')
    # print("Modelo salvo em 'modelo_treinado.pth'")


if __name__ == "__main__":
    main()

