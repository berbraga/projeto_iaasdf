"""
Script principal para executar o treinamento e avaliação do modelo de classificação de culturas agrícolas.
"""
import json
import torch
from model_crops import RedeCnnCulturasAgricolas
from data_loader_crops import preparar_datasets
from trainer_crops import treinar_rede
from evaluator_crops import avaliar_modelo, imprimir_resultados
from visualizador import plotar_curvas_treinamento, plotar_curvas_combinadas


def main():
    """
    Função principal que executa o pipeline completo:
    1. Detecta o dispositivo (GPU ou CPU)
    2. Carrega e prepara os dados (treino/validação)
    3. Treina o modelo
    4. Avalia o modelo
    """
    # Configurações
    caminho_dataset = 'Agricultural-crops'
    tamanho_imagem = 224
    imagens_treino = 20
    imagens_validacao = 12
    epochs = 2000
    learning_rate = 0.00001
    batch_size = 64
    num_classes = 30
    
    # Detectar dispositivo
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("="*70)
    print("CLASSIFICAÇÃO DE CULTURAS AGRÍCOLAS")
    print("="*70)
    print(f"\nGPU está {'disponível' if device == 'cuda' else 'NÃO disponível'}")
    print(f"Usando dispositivo: {device}\n")
    
    # Carregar e preparar dados
    print("="*70)
    print("CARREGANDO DADOS")
    print("="*70)
    dataset_treino, dataset_validacao, classes = preparar_datasets(
        caminho_dataset,
        tamanho_imagem=tamanho_imagem,
        imagens_treino=imagens_treino,
        imagens_validacao=imagens_validacao
    )
    
    if dataset_treino is None or dataset_validacao is None:
        print("ERRO: Não foi possível carregar os datasets!")
        return
    
    print(f"\nNúmero de classes: {len(classes)}")
    
    # Criar modelo
    print("\n" + "="*70)
    print("CRIANDO MODELO")
    print("="*70)
    modelo = RedeCnnCulturasAgricolas(num_classes=len(classes))
    modelo = modelo.to(device)
    
    # Contar parâmetros
    total_params = sum(p.numel() for p in modelo.parameters())
    trainable_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    print(f"Total de parâmetros: {total_params:,}")
    print(f"Parâmetros treináveis: {trainable_params:,}")
    print(f"Modelo criado e movido para {device}")
    
    # Treinar modelo
    print("\n" + "="*70)
    print("TREINANDO MODELO")
    print("="*70)
    modelo_treinado, historico = treinar_rede(
        modelo,
        dataset_treino,
        dataset_validacao,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        device=device
    )
    
    # Avaliar modelo no conjunto de validação
    print("\n" + "="*70)
    print("AVALIANDO MODELO")
    print("="*70)
    resultados = avaliar_modelo(
        modelo_treinado,
        dataset_validacao,
        classes,
        device=device,
        batch_size=batch_size
    )
    imprimir_resultados(resultados, classes)
    
    # Gerar gráficos das curvas de treinamento
    print("\n" + "="*70)
    print("GERANDO GRÁFICOS")
    print("="*70)
    plotar_curvas_treinamento(historico, 'curvas_treinamento.png')
    plotar_curvas_combinadas(historico, 'curvas_treinamento_combinadas.png')
    
    # Salvar modelo final
    torch.save(modelo_treinado.state_dict(), 'modelo_final_culturas.pth')
    print(f"\nModelo salvo em 'modelo_final_culturas.pth'")
    
    # Salvar lista de classes
    with open('classes_culturas.txt', 'w', encoding='utf-8') as f:
        for i, classe in enumerate(classes):
            f.write(f"{i}: {classe}\n")
    print(f"Lista de classes salva em 'classes_culturas.txt'")


if __name__ == "__main__":
    main()

