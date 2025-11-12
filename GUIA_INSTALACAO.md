# 📦 Guia de Instalação do Projeto

Este guia explica passo a passo como instalar e configurar o projeto de classificação de imagens.

## 📋 Pré-requisitos

Antes de começar, você precisa ter:

- **Python 3.7 ou superior** instalado
- **pip** (gerenciador de pacotes Python)
- **Git** (opcional, apenas se for clonar do repositório)

### Verificar Python

Abra o terminal (PowerShell no Windows, Terminal no Linux/Mac) e execute:

```bash
python --version
```

ou

```bash
python3 --version
```

Se não tiver Python instalado, baixe em: https://www.python.org/downloads/

## 🚀 Instalação Passo a Passo

### Passo 1: Navegar até a Pasta do Projeto

Abra o terminal e navegue até a pasta do projeto:

```bash
cd caminho/para/projeto_iaasdf
```

**No Windows:**
```bash
cd C:\Users\bernardo\Documents\faculdade\projeto_iaasdf
```

### Passo 2: Criar Ambiente Virtual (Recomendado)

Criar um ambiente virtual isola as dependências do projeto e evita conflitos.

#### Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux/Mac:
```bash
python3 -m venv venv
source venv/bin/activate
```

**Como saber se está ativado?** Você verá `(venv)` no início da linha do terminal.

### Passo 3: Instalar Dependências

#### Opção A: Instalação com GPU (Recomendado se tiver GPU NVIDIA)

Se você tem uma GPU NVIDIA e quer usar GPU para acelerar o treinamento:

```bash
pip install -r requirements.txt
```

**Nota:** Se você não tiver CUDA instalado, o PyTorch será instalado na versão CPU automaticamente.

#### Opção B: Instalação apenas CPU (Mais leve)

Se você não tem GPU ou quer instalar apenas a versão CPU:

```bash
pip install -r requirements-cpu.txt
```

#### Opção C: Instalação Manual com GPU

Se você tem CUDA instalado e quer especificar a versão:

**Para CUDA 11.8:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install pillow numpy jupyter scikit-learn
```

**Para CUDA 12.1:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install pillow numpy jupyter scikit-learn
```

### Passo 4: Verificar Instalação

Execute o script de verificação para confirmar que tudo está instalado corretamente:

```bash
python verificar_instalacao.py
```

Este script verifica:
- ✅ Versão do PyTorch instalada
- ✅ Se CUDA/GPU está disponível
- ✅ Informações da GPU (se disponível)
- ✅ Teste de operação na GPU

### Passo 5: Verificar Arquivos de Dados

Certifique-se de que os arquivos necessários estão presentes:

#### Para Classificação de Pássaros (`main.py`):
- ✅ `bird.zip` - arquivo ZIP com imagens de pássaros
- ✅ `not-bird.zip` - arquivo ZIP com imagens de não-pássaros

#### Para Classificação de Culturas (`main_crops.py`):
- ✅ `Agricultural-crops/` - pasta com 30 subpastas de culturas agrícolas

## ✅ Testar a Instalação

Execute o script de teste para verificar se tudo está funcionando:

```bash
python testar_projeto.py
```

Este script testa:
- ✅ Importação de módulos
- ✅ Criação do modelo
- ✅ Dependências instaladas
- ✅ Dispositivo (CPU/GPU)

## 🎯 Executar o Projeto

### Opção 1: Classificação de Pássaros

```bash
python main.py
```

### Opção 2: Classificação de Culturas Agrícolas

```bash
python main_crops.py
```

### Opção 3: Usar Jupyter Notebook

```bash
jupyter notebook
```

Depois abra o arquivo `image.ipynb` ou `kernels.ipynb` no navegador.

## 🔧 Solução de Problemas

### Problema: "pip não é reconhecido"

**Solução:**
- Certifique-se de que Python está instalado corretamente
- Use `python -m pip` em vez de apenas `pip`
- No Windows, reinstale Python marcando "Add Python to PATH"

### Problema: "ModuleNotFoundError"

**Solução:**
```bash
# Reinstalar dependências
pip install -r requirements.txt

# Ou instalar manualmente
pip install torch torchvision pillow numpy jupyter scikit-learn
```

### Problema: "CUDA out of memory" (durante treinamento)

**Solução:**
1. Reduza o `batch_size` no arquivo `main.py` ou `main_crops.py`
2. Reduza o número de imagens carregadas
3. Feche outros programas que usam GPU

### Problema: GPU não detectada

**Solução:**
1. Verifique se tem GPU NVIDIA: `nvidia-smi` (no terminal)
2. Instale CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
3. Instale PyTorch com CUDA: veja Passo 3 - Opção C
4. Execute `python verificar_instalacao.py` para verificar

### Problema: Ambiente virtual não ativa

**Windows:**
```bash
# Se o comando acima não funcionar, tente:
.\venv\Scripts\Activate.ps1

# Se der erro de política, execute no PowerShell como administrador:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Linux/Mac:**
```bash
# Certifique-se de usar o caminho correto
source ./venv/bin/activate
```

## 📝 Estrutura do Projeto

Após a instalação, sua estrutura deve estar assim:

```
projeto_iaasdf/
├── venv/                    # Ambiente virtual (criado por você)
├── main.py                  # Script principal - Classificação de pássaros
├── main_crops.py            # Script principal - Classificação de culturas
├── model.py                 # Modelo CNN para pássaros
├── model_crops.py           # Modelo CNN para culturas
├── data_loader.py           # Carregamento de dados (pássaros)
├── data_loader_crops.py     # Carregamento de dados (culturas)
├── trainer.py               # Função de treinamento (pássaros)
├── trainer_crops.py         # Função de treinamento (culturas)
├── evaluator.py             # Avaliação do modelo (pássaros)
├── evaluator_crops.py       # Avaliação do modelo (culturas)
├── requirements.txt         # Dependências (com GPU)
├── requirements-cpu.txt     # Dependências (apenas CPU)
├── verificar_instalacao.py  # Script de verificação
├── testar_projeto.py        # Script de teste
├── GUIA_GPU.md             # Guia de uso de GPU
├── GUIA_INSTALACAO.md      # Este arquivo
├── README.md               # Documentação principal
└── [seus dados]            # bird.zip, not-bird.zip, Agricultural-crops/
```

## 🎓 Próximos Passos

1. **Ler a documentação:**
   - `README.md` - Visão geral do projeto
   - `GUIA_GPU.md` - Como usar GPU (se tiver)
   - `COMO_TESTAR.md` - Como testar o projeto

2. **Executar o projeto:**
   ```bash
   python main.py
   ```

3. **Ajustar configurações:**
   - Edite `main.py` ou `main_crops.py` para alterar parâmetros
   - Ajuste `batch_size`, `epochs`, `learning_rate` conforme necessário

## 💡 Dicas

- **Use ambiente virtual:** Sempre recomendado para isolar dependências
- **Monitore GPU:** Se usar GPU, monitore o uso com `nvidia-smi` (Linux) ou Task Manager (Windows)
- **Comece pequeno:** Teste com poucas épocas primeiro para verificar se está funcionando
- **Salve modelos:** Descomente as linhas de salvamento no código para salvar modelos treinados

## 📚 Recursos Adicionais

- [Documentação PyTorch](https://pytorch.org/docs/)
- [Guia de Instalação PyTorch](https://pytorch.org/get-started/locally/)
- [Documentação CUDA](https://docs.nvidia.com/cuda/)

## ❓ Precisa de Ajuda?

Se encontrar problemas:

1. Execute `python verificar_instalacao.py` para diagnóstico
2. Execute `python testar_projeto.py` para testes
3. Verifique os logs de erro no terminal
4. Consulte a seção "Solução de Problemas" acima

---

**Pronto!** Agora você está pronto para usar o projeto. 🎉


