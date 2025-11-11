# 🧪 Como Testar o Projeto

Este guia explica como testar se o projeto está funcionando e como classificar imagens.

## 📋 Passo 1: Testar se o Projeto Funciona

Antes de treinar o modelo completo, você pode fazer um teste rápido:

```bash
# Ativar ambiente virtual (se estiver usando)
source venv/bin/activate

# Executar testes
python testar_projeto.py
```

Este script verifica:
- ✅ Se todas as dependências estão instaladas
- ✅ Se o dispositivo (CPU/GPU) está funcionando
- ✅ Se o modelo pode ser criado
- ✅ Se os dados podem ser carregados

**Resultado esperado:** Todos os testes devem passar ✅

---

## 🚀 Passo 2: Treinar o Modelo

Depois que os testes passarem, treine o modelo completo:

```bash
python main_crops.py
```

Isso vai:
1. Carregar todas as imagens (20 treino + 12 validação por classe)
2. Treinar o modelo por 50 épocas
3. Salvar o melhor modelo em `modelo_final_culturas.pth`
4. Gerar relatório de avaliação

**Tempo estimado:**
- CPU: 30-60 minutos
- GPU: 5-10 minutos

---

## 🖼️ Passo 3: Classificar uma Imagem

Depois de treinar o modelo, você pode classificar qualquer imagem:

### Opção 1: Usar uma imagem do dataset

```bash
# Exemplo: classificar uma imagem de banana
python classificar_imagem.py Agricultural-crops/banana/image\ \(1\).jpg
```

### Opção 2: Usar uma imagem própria

```bash
# Exemplo: classificar uma foto sua
python classificar_imagem.py minha_foto.jpg
```

### Opção 3: Especificar modelo diferente

```bash
python classificar_imagem.py imagem.jpg melhor_modelo_culturas.pth
```

---

## 📊 Exemplo de Saída

Quando você executar `classificar_imagem.py`, verá algo assim:

```
======================================================================
CLASSIFICAÇÃO DE IMAGEM
======================================================================
Imagem: Agricultural-crops/banana/image (1).jpg
Modelo: modelo_final_culturas.pth
Dispositivo: cpu

Carregando modelo...
✅ Modelo carregado com sucesso

Processando imagem...
✅ Imagem processada

Classificando...

======================================================================
RESULTADOS DA CLASSIFICAÇÃO
======================================================================

1. banana                          85.23% ████████████████████████
2. papaya                           8.45% ████
3. Coconut                          3.12% ██
4. Lemon                            1.89% █
5. tomato                           0.31% 

======================================================================
PREDIÇÃO: banana
CONFIANÇA: 85.23%
======================================================================
✅ Alta confiança na predição
```

---

## 🔍 Verificando se uma Imagem é de uma Cultura

O modelo classifica imagens em **30 classes de culturas agrícolas**. Para verificar se uma imagem é de uma cultura específica:

### Exemplo 1: Verificar se é banana

```bash
python classificar_imagem.py minha_imagem.jpg
```

Se a primeira predição for "banana" com alta confiança (>50%), provavelmente é uma banana.

### Exemplo 2: Verificar se é uma das culturas do dataset

O modelo reconhece estas 30 culturas:
- almond, banana, cardamom, Cherry, chilli, clove, coconut
- Coffee-plant, cotton, Cucumber, Fox_nut(Makhana), gram, jowar, jute
- Lemon, maize, mustard-oil, Olive-tree, papaya, Pearl_millet(bajra)
- pineapple, rice, soyabean, sugarcane, sunflower, tea
- Tobacco-plant, tomato, vigna-radiati(Mung), wheat

Se a imagem não for nenhuma dessas culturas, o modelo pode:
- ❌ Dar baixa confiança (<30%)
- ⚠️ Classificar incorretamente como a cultura mais similar

---

## ⚠️ Problemas Comuns

### 1. "Modelo não encontrado"

**Erro:**
```
❌ ERRO: Modelo não encontrado em 'modelo_final_culturas.pth'
```

**Solução:** Você precisa treinar o modelo primeiro:
```bash
python main_crops.py
```

### 2. "Imagem não encontrada"

**Erro:**
```
❌ ERRO: Imagem não encontrada: minha_imagem.jpg
```

**Solução:** Verifique se o caminho da imagem está correto. Use caminho absoluto ou relativo.

### 3. "Dataset não encontrado"

**Erro:**
```
❌ ERRO: Pasta 'Agricultural-crops' não encontrada!
```

**Solução:** Certifique-se de que a pasta `Agricultural-crops` está no mesmo diretório do script.

---

## 🎯 Dicas para Melhores Resultados

1. **Use imagens claras** - O modelo funciona melhor com imagens bem iluminadas
2. **Imagens focadas** - Evite imagens borradas
3. **Tamanho adequado** - O modelo redimensiona para 224x224, mas imagens maiores geralmente são melhores
4. **Culturas do dataset** - O modelo foi treinado apenas nas 30 culturas do dataset

---

## 📝 Resumo Rápido

```bash
# 1. Testar projeto
python testar_projeto.py

# 2. Treinar modelo
python main_crops.py

# 3. Classificar imagem
python classificar_imagem.py imagem.jpg
```

---

## ❓ FAQ

**P: O modelo pode classificar imagens que não estão no dataset?**
R: Sim, mas com menor precisão. O modelo foi treinado apenas nas 30 culturas do dataset.

**P: Como saber se a classificação está correta?**
R: Verifique a confiança (probabilidade). Se for >50%, geralmente está correto. Se for <30%, pode estar errado.

**P: Posso usar o modelo sem treinar?**
R: Não. Você precisa treinar primeiro para gerar o arquivo `modelo_final_culturas.pth`.

**P: Quanto tempo leva para classificar uma imagem?**
R: Menos de 1 segundo em CPU, instantâneo em GPU.

