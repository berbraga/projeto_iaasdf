"""
Script para classificar uma imagem individual usando o modelo treinado.
"""
import torch
from PIL import Image
from torchvision import transforms
from model_crops import RedeCnnCulturasAgricolas
import os
import sys


def carregar_modelo(caminho_modelo='modelo_final_culturas.pth', num_classes=30, device='cpu'):
    """
    Carrega o modelo treinado.
    
    Args:
        caminho_modelo: Caminho para o arquivo do modelo
        num_classes: Número de classes
        device: Dispositivo ('cpu' ou 'cuda')
        
    Returns:
        Modelo carregado
    """
    if not os.path.exists(caminho_modelo):
        print(f"❌ ERRO: Modelo não encontrado em '{caminho_modelo}'")
        print("   Primeiro você precisa treinar o modelo executando: python main_crops.py")
        return None
    
    modelo = RedeCnnCulturasAgricolas(num_classes=num_classes)
    modelo.load_state_dict(torch.load(caminho_modelo, map_location=device))
    modelo = modelo.to(device)
    modelo.eval()
    
    return modelo


def carregar_classes(caminho_classes='classes_culturas.txt'):
    """
    Carrega a lista de classes do arquivo.
    
    Args:
        caminho_classes: Caminho para o arquivo com as classes
        
    Returns:
        Lista de nomes das classes
    """
    if not os.path.exists(caminho_classes):
        print(f"⚠️  Aviso: Arquivo de classes não encontrado. Usando índices numéricos.")
        return [f"Classe {i}" for i in range(30)]
    
    classes = []
    with open(caminho_classes, 'r', encoding='utf-8') as f:
        for linha in f:
            linha = linha.strip()
            if ':' in linha:
                _, nome_classe = linha.split(':', 1)
                classes.append(nome_classe.strip())
    
    return classes


def preprocessar_imagem(caminho_imagem, tamanho=224, normalizar=True):
    """
    Carrega e preprocessa uma imagem para classificação.
    
    Args:
        caminho_imagem: Caminho para a imagem
        tamanho: Tamanho para redimensionar (padrão: 224)
        normalizar: Se True, aplica normalização estatística (deve ser igual ao treinamento)
        
    Returns:
        Tensor da imagem processada
    """
    transformacoes = [
        transforms.Resize((tamanho, tamanho)),
        transforms.ToTensor()  # Converte para [0, 1]
    ]
    
    # Normalização estatística (mesma usada no treinamento)
    if normalizar:
        transformacoes.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # Média para cada canal RGB
                std=[0.229, 0.224, 0.225]     # Desvio padrão para cada canal RGB
            )
        )
    
    transform = transforms.Compose(transformacoes)
    
    try:
        imagem = Image.open(caminho_imagem).convert('RGB')
        tensor = transform(imagem)
        return tensor.unsqueeze(0)  # Adiciona dimensão do batch
    except Exception as e:
        print(f"❌ ERRO ao carregar imagem: {e}")
        return None


def classificar_imagem(caminho_imagem, caminho_modelo='modelo_final_culturas.pth', 
                       top_k=5, device=None):
    """
    Classifica uma imagem e retorna as classes mais prováveis.
    
    Args:
        caminho_imagem: Caminho para a imagem a ser classificada
        caminho_modelo: Caminho para o modelo treinado
        top_k: Número de top classes para mostrar
        device: Dispositivo ('cpu' ou 'cuda'), None para auto-detectar
        
    Returns:
        Lista de tuplas (classe, probabilidade)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*70)
    print("CLASSIFICAÇÃO DE IMAGEM")
    print("="*70)
    print(f"Imagem: {caminho_imagem}")
    print(f"Modelo: {caminho_modelo}")
    print(f"Dispositivo: {device}\n")
    
    # Carregar classes
    classes = carregar_classes()
    
    # Carregar modelo
    print("Carregando modelo...")
    modelo = carregar_modelo(caminho_modelo, num_classes=len(classes), device=device)
    if modelo is None:
        return None
    
    print("✅ Modelo carregado com sucesso\n")
    
    # Preprocessar imagem
    print("Processando imagem...")
    imagem_tensor = preprocessar_imagem(caminho_imagem)
    if imagem_tensor is None:
        return None
    
    imagem_tensor = imagem_tensor.to(device)
    print("✅ Imagem processada\n")
    
    # Classificar
    print("Classificando...")
    with torch.no_grad():
        outputs = modelo(imagem_tensor)
        probabilidades = torch.softmax(outputs, dim=1)
        prob, indices = torch.topk(probabilidades, top_k)
    
    # Converter para CPU e numpy
    prob = prob.cpu().numpy()[0]
    indices = indices.cpu().numpy()[0]
    
    # Criar lista de resultados
    resultados = []
    for i, idx in enumerate(indices):
        nome_classe = classes[idx] if idx < len(classes) else f"Classe {idx}"
        resultados.append((nome_classe, prob[i] * 100))
    
    return resultados


def imprimir_resultados(resultados):
    """Imprime os resultados da classificação de forma formatada."""
    if resultados is None:
        return
    
    print("="*70)
    print("RESULTADOS DA CLASSIFICAÇÃO")
    print("="*70)
    print()
    
    for i, (classe, probabilidade) in enumerate(resultados, 1):
        barra = "█" * int(probabilidade / 2)  # Barra visual
        print(f"{i}. {classe:<30} {probabilidade:>6.2f}% {barra}")
    
    print()
    classe_predita = resultados[0][0]
    confianca = resultados[0][1]
    
    print("="*70)
    print(f"PREDIÇÃO: {classe_predita}")
    print(f"CONFIANÇA: {confianca:.2f}%")
    print("="*70)
    
    if confianca > 50:
        print("✅ Alta confiança na predição")
    elif confianca > 30:
        print("⚠️  Confiança moderada na predição")
    else:
        print("❌ Baixa confiança - o modelo não está certo")


def main():
    """Função principal."""
    if len(sys.argv) < 2:
        print("Uso: python classificar_imagem.py <caminho_da_imagem> [caminho_do_modelo]")
        print("\nExemplos:")
        print("  python classificar_imagem.py imagem.jpg")
        print("  python classificar_imagem.py imagem.jpg modelo_final_culturas.pth")
        print("\nNota: Você precisa treinar o modelo primeiro executando:")
        print("  python main_crops.py")
        sys.exit(1)
    
    caminho_imagem = sys.argv[1]
    caminho_modelo = sys.argv[2] if len(sys.argv) > 2 else 'modelo_final_culturas.pth'
    
    if not os.path.exists(caminho_imagem):
        print(f"❌ ERRO: Imagem não encontrada: {caminho_imagem}")
        sys.exit(1)
    
    resultados = classificar_imagem(caminho_imagem, caminho_modelo)
    imprimir_resultados(resultados)


if __name__ == "__main__":
    main()

