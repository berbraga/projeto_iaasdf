# 🚀 Guia de Uso de GPU

Este guia explica como configurar e usar GPU para acelerar o treinamento dos modelos de classificação.

## 📋 Pré-requisitos

### 1. Verificar se sua GPU suporta CUDA

Para usar GPU com PyTorch, você precisa de uma GPU NVIDIA com suporte a CUDA. Verifique se sua GPU é compatível visitando: https://developer.nvidia.com/cuda-gpus

### 2. Instalar CUDA Toolkit

1. **Baixar CUDA Toolkit**: Acesse https://developer.nvidia.com/cuda-downloads
2. **Instalar**: Siga as instruções para sua plataforma (Windows/Linux)
3. **Verificar instalação**: Abra o terminal e execute:
   ```bash
   nvcc --version
   ```

### 3. Instalar PyTorch com suporte CUDA

O projeto já está configurado para detectar automaticamente GPU. Você só precisa instalar o PyTorch com suporte CUDA.

#### Opção A: Instalação via pip (Recomendado)

**Para CUDA 11.8:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Para CUDA 12.1:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Para CUDA 12.4:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

#### Opção B: Instalação via requirements.txt

O arquivo `requirements.txt` já instala PyTorch com suporte CUDA por padrão. Basta executar:

```bash
pip install -r requirements.txt
```

**Nota**: Se você não tiver CUDA instalado, o PyTorch será instalado na versão CPU. Para forçar instalação CPU, use:
```bash
pip install -r requirements-cpu.txt
```

## ✅ Verificar se GPU está disponível

Crie um script de teste ou execute no Python:

```python
import torch

print(f"PyTorch versão: {torch.__version__}")
print(f"CUDA disponível: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Versão CUDA: {torch.version.cuda}")
    print(f"Nome da GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memória GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("⚠️  GPU não disponível. O treinamento usará CPU.")
```

Ou execute o script de verificação do projeto:

```bash
python verificar_instalacao.py
```

## 🎯 Como o Projeto Usa GPU

O projeto **detecta automaticamente** se há GPU disponível e a usa quando possível. Você não precisa fazer nenhuma configuração adicional!

### Detecção Automática

Ambos os scripts principais (`main.py` e `main_crops.py`) já fazem a detecção:

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"GPU está {'disponível' if device == 'cuda' else 'NÃO disponível'}")
print(f"Usando dispositivo: {device}\n")
```

### O que é movido para GPU

- ✅ **Modelo**: O modelo é movido para GPU com `.to(device)`
- ✅ **Dados de treinamento**: Os batches são movidos para GPU durante o treinamento
- ✅ **Dados de validação**: Os batches são movidos para GPU durante a validação

## 🚀 Executando o Projeto com GPU

### 1. Classificação de Pássaros

```bash
python main.py
```

O script detectará automaticamente a GPU e usará se disponível.

### 2. Classificação de Culturas Agrícolas

```bash
python main_crops.py
```

O script detectará automaticamente a GPU e usará se disponível.

## 📊 Verificando o Uso de GPU Durante Treinamento

### Windows (Task Manager)

1. Abra o **Gerenciador de Tarefas** (Ctrl + Shift + Esc)
2. Vá para a aba **Desempenho**
3. Selecione sua GPU
4. Monitore o uso durante o treinamento

### Linux (nvidia-smi)

Execute em um terminal separado:

```bash
watch -n 1 nvidia-smi
```

Isso atualizará a cada segundo mostrando:
- Uso de memória GPU
- Utilização da GPU (%)
- Processos em execução

### Python (Durante execução)

Adicione este código no seu script para monitorar:

```python
import torch

if torch.cuda.is_available():
    print(f"Memória GPU alocada: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"Memória GPU reservada: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
```

## ⚙️ Otimizações para GPU

### Aumentar Batch Size

Com GPU, você pode aumentar o `batch_size` para acelerar o treinamento:

**Em `main.py`:**
```python
batch_size = 128  # Aumente de 64 para 128 ou mais (depende da memória GPU)
```

**Em `main_crops.py`:**
```python
batch_size = 64  # Aumente de 32 para 64 ou mais (depende da memória GPU)
```

**Atenção**: Aumente gradualmente e monitore o uso de memória. Se der erro de "out of memory", reduza o batch_size.

### Usar Mixed Precision (Opcional)

Para GPUs modernas (Tensor Cores), você pode usar precisão mista para acelerar ainda mais:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# No loop de treinamento:
with autocast():
    outputs = modelo(inputs)
    loss = criterio(outputs, targets)

scaler.scale(loss).backward()
scaler.step(otimizador)
scaler.update()
```

## 🐛 Solução de Problemas

### Problema: "CUDA out of memory"

**Solução:**
1. Reduza o `batch_size` no arquivo de configuração
2. Reduza o número de imagens carregadas
3. Feche outros programas que usam GPU

### Problema: GPU não é detectada

**Verificações:**
1. ✅ CUDA Toolkit instalado?
2. ✅ PyTorch com suporte CUDA instalado?
3. ✅ Drivers NVIDIA atualizados?
4. ✅ GPU compatível com CUDA?

**Teste:**
```python
import torch
print(torch.cuda.is_available())  # Deve retornar True
```

### Problema: Treinamento mais lento na GPU

**Possíveis causas:**
1. Dataset muito pequeno (overhead de transferência CPU→GPU)
2. Batch size muito pequeno
3. GPU antiga ou com pouca memória

**Solução:** Para datasets pequenos, CPU pode ser mais rápida. Use GPU para datasets maiores.

## 📝 Notas Importantes

- ⚠️ **Memória GPU**: Monitore o uso de memória. Se exceder, reduza o batch_size
- ⚠️ **Compatibilidade**: Certifique-se de que a versão do CUDA Toolkit corresponde à versão do PyTorch
- ✅ **Fallback automático**: Se GPU não estiver disponível, o projeto usa CPU automaticamente
- ✅ **Sem configuração extra**: O projeto já está configurado para usar GPU quando disponível

## 🔗 Links Úteis

- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [CUDA Toolkit Downloads](https://developer.nvidia.com/cuda-downloads)
- [NVIDIA GPU Compatibility](https://developer.nvidia.com/cuda-gpus)


