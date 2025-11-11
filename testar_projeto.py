"""
Script para testar rapidamente se o projeto está funcionando corretamente.
Executa um teste rápido sem treinar o modelo completo.
"""
import torch
from model_crops import RedeCnnCulturasAgricolas
from data_loader_crops import preparar_datasets
import os


def testar_carregamento_dados():
    """Testa se os dados são carregados corretamente."""
    print("="*70)
    print("TESTE 1: Carregamento de Dados")
    print("="*70)
    
    if not os.path.exists('Agricultural-crops'):
        print("❌ ERRO: Pasta 'Agricultural-crops' não encontrada!")
        return False
    
    try:
        dataset_treino, dataset_validacao, classes = preparar_datasets(
            'Agricultural-crops',
            tamanho_imagem=224,
            imagens_treino=2,  # Apenas 2 para teste rápido
            imagens_validacao=1  # Apenas 1 para teste rápido
        )
        
        if dataset_treino is None or dataset_validacao is None:
            print("❌ ERRO: Não foi possível criar os datasets!")
            return False
        
        print(f"✅ Sucesso! {len(classes)} classes encontradas")
        print(f"✅ Dataset treino: {len(dataset_treino)} imagens")
        print(f"✅ Dataset validação: {len(dataset_validacao)} imagens")
        return True
        
    except Exception as e:
        print(f"❌ ERRO ao carregar dados: {e}")
        return False


def testar_modelo():
    """Testa se o modelo pode ser criado e executar forward pass."""
    print("\n" + "="*70)
    print("TESTE 2: Criação e Teste do Modelo")
    print("="*70)
    
    try:
        # Criar modelo
        modelo = RedeCnnCulturasAgricolas(num_classes=30)
        print("✅ Modelo criado com sucesso")
        
        # Contar parâmetros
        total_params = sum(p.numel() for p in modelo.parameters())
        print(f"✅ Total de parâmetros: {total_params:,}")
        
        # Testar forward pass
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        modelo = modelo.to(device)
        
        # Criar tensor de teste (batch de 2 imagens 224x224)
        x_teste = torch.randn(2, 3, 224, 224).to(device)
        
        modelo.eval()
        with torch.no_grad():
            output = modelo(x_teste)
        
        print(f"✅ Forward pass executado com sucesso")
        print(f"✅ Input shape: {x_teste.shape}")
        print(f"✅ Output shape: {output.shape}")
        print(f"✅ Output esperado: (2, 30) - ✓ Correto!")
        
        return True
        
    except Exception as e:
        print(f"❌ ERRO ao testar modelo: {e}")
        import traceback
        traceback.print_exc()
        return False


def testar_dependencias():
    """Testa se todas as dependências estão instaladas."""
    print("\n" + "="*70)
    print("TESTE 3: Verificação de Dependências")
    print("="*70)
    
    dependencias = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
        'sklearn': 'scikit-learn'
    }
    
    todas_ok = True
    for modulo, nome in dependencias.items():
        try:
            __import__(modulo)
            print(f"✅ {nome} instalado")
        except ImportError:
            print(f"❌ {nome} NÃO instalado! Execute: pip install {nome.lower()}")
            todas_ok = False
    
    return todas_ok


def testar_dispositivo():
    """Testa se o dispositivo (CPU/GPU) está funcionando."""
    print("\n" + "="*70)
    print("TESTE 4: Verificação de Dispositivo")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Dispositivo detectado: {device}")
    
    if device == 'cuda':
        print(f"✅ GPU disponível: {torch.cuda.get_device_name(0)}")
    else:
        print("ℹ️  Usando CPU (GPU não disponível)")
    
    # Testar operação simples
    try:
        x = torch.randn(10, 10).to(device)
        y = torch.randn(10, 10).to(device)
        z = torch.matmul(x, y)
        print(f"✅ Operações no {device} funcionando corretamente")
        return True
    except Exception as e:
        print(f"❌ ERRO nas operações: {e}")
        return False


def main():
    """Executa todos os testes."""
    print("\n" + "="*70)
    print("TESTES DO PROJETO - CLASSIFICAÇÃO DE CULTURAS AGRÍCOLAS")
    print("="*70)
    print()
    
    resultados = []
    
    # Teste 1: Dependências
    resultados.append(("Dependências", testar_dependencias()))
    
    # Teste 2: Dispositivo
    resultados.append(("Dispositivo", testar_dispositivo()))
    
    # Teste 3: Modelo
    resultados.append(("Modelo", testar_modelo()))
    
    # Teste 4: Carregamento de dados
    resultados.append(("Carregamento de Dados", testar_carregamento_dados()))
    
    # Resumo
    print("\n" + "="*70)
    print("RESUMO DOS TESTES")
    print("="*70)
    
    for nome, resultado in resultados:
        status = "✅ PASSOU" if resultado else "❌ FALHOU"
        print(f"{nome}: {status}")
    
    todos_passaram = all(r[1] for r in resultados)
    
    print("\n" + "="*70)
    if todos_passaram:
        print("✅ TODOS OS TESTES PASSARAM! O projeto está pronto para uso.")
        print("\nPróximo passo: Execute 'python main_crops.py' para treinar o modelo")
    else:
        print("❌ ALGUNS TESTES FALHARAM. Verifique os erros acima.")
    print("="*70)


if __name__ == "__main__":
    main()

