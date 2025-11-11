# Projeto IA - Classificação de Culturas Agrícolas

Este projeto implementa uma rede neural convolucional (CNN) para classificar imagens de 30 diferentes tipos de culturas agrícolas usando PyTorch.

## 📋 Objetivos do Projeto

- Classificar imagens em **30 classes** de plantas agrícolas
- Usar **20 imagens por classe** para treinamento
- Usar **12 imagens por classe** para validação
- Padronizar o tamanho das imagens para 224x224 pixels

## 📁 Estrutura do Dataset

O dataset `Agricultural-crops` contém 30 pastas, cada uma representando uma classe de cultura:
- almond, banana, cardamom, Cherry, chilli, clove, coconut, Coffee-plant, cotton, Cucumber
- Fox_nut(Makhana), gram, jowar, jute, Lemon, maize, mustard-oil, Olive-tree, papaya
- Pearl_millet(bajra), pineapple, rice, soyabean, sugarcane, sunflower, tea
- Tobacco-plant, tomato, vigna-radiati(Mung), wheat

## 🚀 Como Executar

### 1. Instalar as Dependências

```bash
pip install -r requirements.txt
```

Ou, se preferir usar um ambiente virtual:

```bash
# Criar ambiente virtual
python3 -m venv venv

# Ativar ambiente virtual (Linux/Mac)
source venv/bin/activate

# Ativar ambiente virtual (Windows)
venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt
```

### 2. Verificar o Dataset

Certifique-se de que a pasta `Agricultural-crops` está presente no diretório do projeto com as 30 pastas de classes.

### 3. Executar o Script Principal

```bash
python main_crops.py
```

O script executará automaticamente:
1. Carregamento e processamento das imagens (20 treino + 12 validação por classe)
2. Criação do modelo CNN
3. Treinamento do modelo com validação
4. Avaliação do modelo com métricas detalhadas

## 📊 Arquitetura do Modelo

O modelo `RedeCnnCulturasAgricolas` possui:
- **3 camadas convolucionais** (32, 64, 128 filtros) com Batch Normalization
- **3 camadas de MaxPooling** (2x2)
- **1 camada de Adaptive Average Pooling** (para garantir tamanho fixo)
- **2 camadas lineares** (512 neurônios + 30 classes de saída)
- **Dropout** (0.5) para regularização

## 🔧 Configurações

Para ajustar os parâmetros, edite o arquivo `main_crops.py`:

```python
tamanho_imagem = 224          # Tamanho para redimensionar imagens
imagens_treino = 20          # Imagens por classe para treino
imagens_validacao = 12       # Imagens por classe para validação
epochs = 50                  # Número de épocas
learning_rate = 0.001        # Taxa de aprendizado
batch_size = 32              # Tamanho do lote
```

## 📈 Saída Esperada

Durante o treinamento, você verá:
- Progresso do carregamento de imagens por classe
- Métricas de treino e validação por época (Loss e Acurácia)
- Melhor modelo salvo automaticamente

Após o treinamento:
- Matriz de confusão
- Relatório detalhado por classe (Precisão, Recall, F1-Score)
- Acurácia geral e média

## 💾 Arquivos Gerados

Após a execução, serão criados:
- `melhor_modelo_culturas.pth` - Melhor modelo durante o treinamento
- `modelo_final_culturas.pth` - Modelo final após treinamento
- `classes_culturas.txt` - Lista de classes com seus índices

## ⚠️ Notas Importantes

- O projeto detecta automaticamente se há GPU disponível
- Se não houver GPU, o treinamento será executado na CPU (mais lento)
- As imagens são normalizadas usando valores ImageNet (padrão)
- O modelo usa CrossEntropyLoss para classificação multi-classe
- O melhor modelo é salvo automaticamente baseado na acurácia de validação

## 📝 Estrutura de Arquivos

```
projeto_ia/
├── main_crops.py              # Script principal
├── model_crops.py             # Definição da CNN
├── data_loader_crops.py        # Carregamento de dados
├── trainer_crops.py            # Função de treinamento
├── evaluator_crops.py          # Avaliação e métricas
├── Agricultural-crops/         # Dataset com 30 classes
├── requirements.txt           # Dependências
└── README_CROPS.md            # Este arquivo
```

