# Projeto IA - Classificação de Pássaros

Este projeto implementa uma rede neural convolucional (CNN) para classificar imagens entre "pássaro" e "não-pássaro" usando PyTorch.

## 📋 Pré-requisitos

- Python 3.7 ou superior
- CUDA (opcional, para usar GPU)

## 🚀 Instalação Rápida

📖 **Para um guia completo de instalação passo a passo, veja:** [`GUIA_INSTALACAO.md`](GUIA_INSTALACAO.md)

### Instalação Básica

1. **Criar e ativar ambiente virtual (recomendado):**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

2. **Instalar dependências:**

```bash
# Com suporte GPU (recomendado se tiver GPU NVIDIA)
pip install -r requirements.txt

# Apenas CPU (mais leve)
pip install -r requirements-cpu.txt
```

3. **Verificar instalação:**

```bash
python verificar_instalacao.py
```

## 🚀 Como Executar

### Opção 1: Executar como Script Python (Recomendado)

#### 2. Verificar os Arquivos de Dados

Certifique-se de que os seguintes arquivos estão presentes no diretório do projeto:
- `bird.zip` - arquivo ZIP contendo imagens de pássaros
- `not-bird.zip` - arquivo ZIP contendo imagens de não-pássaros

#### 3. Executar o Script Principal

```bash
python main.py
```

O script executará automaticamente:
1. Carregamento e processamento das imagens
2. Criação do modelo
3. Treinamento do modelo
4. Avaliação do modelo

### Opção 2: Executar o Notebook Jupyter

#### 1. Instalar as Dependências

```bash
pip install -r requirements.txt
```

#### 2. Executar o Jupyter Notebook

```bash
jupyter notebook
```

Ou, se preferir JupyterLab:

```bash
jupyter lab
```

No navegador que abrir, clique em `image.ipynb` para abrir o notebook.

#### 3. Executar as Células

Execute as células do notebook na ordem:

1. **Cell 0**: Carrega e processa as imagens dos arquivos ZIP
2. **Cell 1**: Define a arquitetura da CNN
3. **Cell 2**: Define a função de treinamento
4. **Cell 3**: Treina o modelo
5. **Cell 4**: Avalia o modelo treinado

Você pode executar cada célula individualmente usando `Shift + Enter` ou executar todas usando `Cell > Run All`.

## 📁 Estrutura do Projeto

```
projeto_ia/
├── main.py             # Script principal para executar o pipeline completo
├── model.py            # Definição da rede neural convolucional
├── data_loader.py      # Módulo para carregar e processar imagens
├── trainer.py          # Módulo com função de treinamento
├── evaluator.py        # Módulo para avaliar o modelo
├── image.ipynb         # Notebook original (alternativa ao script Python)
├── kernels.ipynb       # Notebook adicional (se houver)
├── requirements.txt    # Dependências do projeto
├── bird.zip            # Dados de treinamento - pássaros
├── not-bird.zip        # Dados de treinamento - não-pássaros
├── dog.png             # Imagem de exemplo
├── gato.jpeg           # Imagem de exemplo
└── README.md           # Este arquivo
```

## 🔧 Configurações

O projeto detecta automaticamente se há GPU disponível. Se você tiver CUDA instalado, o treinamento será executado na GPU, caso contrário, usará a CPU.

### 🚀 Usando GPU

Para usar GPU, você precisa:

1. **Instalar CUDA Toolkit**: https://developer.nvidia.com/cuda-downloads
2. **Instalar PyTorch com suporte CUDA**:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```
3. **Verificar instalação**:
   ```bash
   python verificar_instalacao.py
   ```

📖 **Guia completo de GPU**: Veja o arquivo `GUIA_GPU.md` para instruções detalhadas.

Para ajustar os parâmetros de treinamento, edite o arquivo `main.py`:

```python
# Configurações
max_imagens_por_classe = 1000  # Número de imagens por classe
epochs = 100                    # Número de épocas de treinamento
learning_rate = 0.000001        # Taxa de aprendizado
batch_size = 64                 # Tamanho do lote
```

## 📊 Saída Esperada

Durante o treinamento, você verá:
- Progresso do carregamento de imagens
- Perda total por época
- Tempo de execução por época

Após o treinamento, a célula de avaliação mostrará os resultados da classificação.

## ⚠️ Notas

- O projeto carrega 1000 imagens de cada classe por padrão
- As imagens são redimensionadas para 32x32 pixels
- O modelo usa uma arquitetura CNN com 3 camadas convolucionais e 2 camadas lineares

