"""
Script para verificar qual versão do PyTorch foi instalada e se CUDA está disponível.
"""
import sys

try:
    import torch
    print("=" * 60)
    print("VERIFICAÇÃO DA INSTALAÇÃO DO PYTORCH")
    print("=" * 60)
    print(f"\nVersão do PyTorch: {torch.__version__}")
    
    # Verificar se é build com CUDA ou CPU
    if '+cu' in torch.__version__ or 'cuda' in torch.__version__.lower():
        print("✓ Build com suporte CUDA (GPU)")
        build_tipo = "CUDA"
    elif '+cpu' in torch.__version__:
        print("✓ Build apenas CPU")
        build_tipo = "CPU"
    else:
        print("? Tipo de build não identificado claramente")
        build_tipo = "Desconhecido"
    
    # Verificar se CUDA está disponível no sistema
    cuda_disponivel = torch.cuda.is_available()
    print(f"\nCUDA disponível no sistema: {'SIM ✓' if cuda_disponivel else 'NÃO ✗'}")
    
    if cuda_disponivel:
        print(f"  - Número de GPUs: {torch.cuda.device_count()}")
        print(f"  - Nome da GPU: {torch.cuda.get_device_name(0)}")
        print(f"  - Versão CUDA: {torch.version.cuda}")
        
        # Informações de memória GPU
        props = torch.cuda.get_device_properties(0)
        memoria_total_gb = props.total_memory / (1024**3)
        print(f"  - Memória GPU total: {memoria_total_gb:.2f} GB")
        
        # Teste rápido de operação na GPU
        try:
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.matmul(x, y)
            del x, y, z
            torch.cuda.empty_cache()
            print("  - ✓ Teste de operação GPU: SUCESSO")
        except Exception as e:
            print(f"  - ✗ Teste de operação GPU: FALHOU ({e})")
    else:
        print("  - O PyTorch usará CPU para processamento")
        if build_tipo == "CUDA":
            print("  - ⚠️  Você tem um build com CUDA, mas CUDA não está configurado no sistema")
            print("  - O PyTorch funcionará normalmente, mas apenas com CPU")
            print("\n  Para usar GPU, você precisa:")
            print("    1. Instalar CUDA Toolkit: https://developer.nvidia.com/cuda-downloads")
            print("    2. Instalar drivers NVIDIA atualizados")
            print("    3. Verificar compatibilidade da GPU: https://developer.nvidia.com/cuda-gpus")
    
    print("\n" + "=" * 60)
    print("RESUMO:")
    print(f"  Build instalado: {build_tipo}")
    print(f"  Hardware disponível: {'GPU' if cuda_disponivel else 'CPU'}")
    print("=" * 60)
    
except ImportError:
    print("ERRO: PyTorch não está instalado!")
    print("Execute: pip install -r requirements.txt")
    sys.exit(1)

