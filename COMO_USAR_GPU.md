# 🚀 Como Usar GPU para Treinar/Testar

## 📋 Verificação Rápida

### 1. Verificar se tem GPU NVIDIA

**Windows:**
```bash
nvidia-smi
```

Se aparecer informações da GPU, você tem GPU NVIDIA! Se der erro, não tem GPU NVIDIA ou drivers não estão instalados.

**Ou verificar no Python:**
```bash
python verificar_instalacao.py
```

## 🔧 Passo a Passo para Usar GPU

### Situação 1: Você TEM GPU NVIDIA

#### Passo 1: Instalar CUDA Toolkit

1. Baixe CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
2. Escolha sua versão do Windows
3. Instale seguindo as instruções

#### Passo 2: Instalar PyTorch com CUDA

**Desinstalar versão CPU atual:**
```bash
source venv/Scripts/activate
pip uninstall torch torchvision -y
```

**Instalar versão com CUDA:**

**Para CUDA 11.8 (mais comum):**
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

#### Passo 3: Verificar Instalação

```bash
python verificar_instalacao.py
```

Deve mostrar:
```
CUDA disponível no sistema: SIM ✓
  - Nome da GPU: [nome da sua GPU]
  - Versão CUDA: [versão]
```

### Situação 2: Você NÃO TEM GPU NVIDIA

O projeto funciona normalmente na CPU, apenas será mais lento:
- **CPU:** 30-60 minutos para treinar
- **GPU:** 5-15 minutos para treinar

## 🎯 Como o Projeto Usa GPU Automaticamente

**Boa notícia:** O projeto já está configurado para usar GPU automaticamente! Você não precisa mudar nada no código.

### Treinamento com GPU

```bash
# Ativar ambiente virtual
source venv/Scripts/activate

# Treinar - vai usar GPU se disponível
python main_crops.py
```

**O que você verá:**
```
GPU está disponível
Usando dispositivo: cuda
```

ou

```
GPU está NÃO disponível
Usando dispositivo: cpu
```

### Teste/Classificação com GPU

```bash
# Classificar imagem - usa GPU automaticamente se disponível
python classificar_imagem.py imagem.jpg
```

## 📊 Verificar se Está Usando GPU Durante Treinamento

### Windows - Task Manager

1. Abra **Gerenciador de Tarefas** (Ctrl + Shift + Esc)
2. Vá na aba **Desempenho**
3. Selecione sua **GPU**
4. Durante o treinamento, você verá:
   - Uso de GPU aumentando
   - Memória GPU sendo usada

### Python - Durante Execução

Adicione este código no início de `main_crops.py` para monitorar:

```python
import torch

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memória GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Monitorar durante treinamento
    print(f"Memória alocada: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
```

## ⚡ Otimizações para GPU

### Aumentar Batch Size

Com GPU, você pode aumentar o `batch_size` para acelerar:

**Editar `main_crops.py` (linha 26):**
```python
batch_size = 64  # Aumentar de 32 para 64 ou mais
```

**Atenção:** Aumente gradualmente. Se der erro "out of memory", reduza.

### Verificar Uso de Memória GPU

```bash
# Em outro terminal, enquanto treina
nvidia-smi -l 1  # Atualiza a cada 1 segundo
```

## 🐛 Solução de Problemas

### Problema: "CUDA out of memory"

**Solução:**
1. Reduza `batch_size` em `main_crops.py`
2. Feche outros programas que usam GPU
3. Reduza número de imagens carregadas

### Problema: GPU não detectada após instalar CUDA

**Verificações:**
1. Reinicie o terminal após instalar CUDA
2. Verifique drivers NVIDIA: `nvidia-smi`
3. Reinstale PyTorch com CUDA
4. Execute: `python verificar_instalacao.py`

### Problema: PyTorch instalado mas GPU não funciona

**Solução:**
```bash
# Desinstalar
pip uninstall torch torchvision -y

# Reinstalar com CUDA específica
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 📝 Resumo Rápido

### Se TEM GPU NVIDIA:

1. **Instalar CUDA Toolkit** (se ainda não tiver)
2. **Instalar PyTorch com CUDA:**
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```
3. **Verificar:**
   ```bash
   python verificar_instalacao.py
   ```
4. **Treinar (usa GPU automaticamente):**
   ```bash
   python main_crops.py
   ```

### Se NÃO TEM GPU:

- O projeto funciona na CPU normalmente
- Apenas será mais lento (30-60 min vs 5-15 min)

## ✅ Checklist

Antes de treinar com GPU, verifique:

- [ ] GPU NVIDIA instalada
- [ ] Drivers NVIDIA atualizados (`nvidia-smi` funciona)
- [ ] CUDA Toolkit instalado
- [ ] PyTorch com CUDA instalado
- [ ] `python verificar_instalacao.py` mostra GPU disponível
- [ ] Ambiente virtual ativado

---

**Lembre-se:** O projeto detecta GPU automaticamente. Se GPU estiver disponível, será usada. Se não, usa CPU. Não precisa mudar código! 🎉


