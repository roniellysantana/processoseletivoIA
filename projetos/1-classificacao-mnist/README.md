# Projeto 1 — Classificação MNIST

## 💻 O Desafio Técnico

Desenvolva um **modelo de Visão Computacional** capaz de **classificar dígitos manuscritos (0-9)**, e posteriormente **otimize-o para execução em dispositivos Edge**.

O foco não é apenas obter alta acurácia, mas **compreender o fluxo completo**:

**treinamento → validação → salvamento → conversão → otimização**

## 🎯 Conjunto de Dados

Dataset **MNIST**, disponível diretamente via `tf.keras.datasets.mnist` (não é necessário download manual).

## ✅ Requisitos Obrigatórios

### Etapa 1 — Treinamento do Modelo (`train_model.py`)

Implemente:

- Carregamento do dataset MNIST via TensorFlow
- **Split explícito treino/validação** (ex: `validation_split` ou um split manual)
- Construção de uma CNN com:
  - **3 a 4 blocos convolucionais** (`Conv2D` + `BatchNormalization` + `MaxPooling2D`)
  - Camada de `Dropout` antes da saída, para regularização
- Treinamento com **early stopping** baseado na perda de validação (`EarlyStopping`)
- Exibição da **acurácia de validação final** no terminal
- Salvamento do modelo treinado em formato Keras (`model.h5`)

### Etapa 2 — Otimização do Modelo (`optimize_model.py`)

Implemente:

- Carregamento do `model.h5` treinado
- Conversão para **TensorFlow Lite** (`model.tflite`)
- Aplicação de uma técnica de otimização (ex: **Dynamic Range Quantization**)

### Etapa 3 — Inferência com o Modelo Otimizado (`run_inference.py`)

Implemente:

- Carregamento especificamente do **`model.tflite`** (o artefato de edge — não
  o `model.h5`) usando `tf.lite.Interpreter`
- Execução de inferência em pelo menos **5 amostras** do conjunto de teste
- Exibição no terminal, para cada amostra, da classe **predita** vs. a classe **real**

> 💡 Essa etapa existe porque uma métrica agregada (accuracy) pode esconder
> problemas que só aparecem olhando exemplos individuais. Também é o teste mais
> próximo do uso real em produção: carregar o artefato de edge e classificar
> uma entrada por vez.

**Objetivo:** reduzir o tamanho do modelo, mantendo desempenho adequado para aplicações de Edge AI.

## 📂 Estrutura da Pasta

⚠️ Não altere os nomes dos arquivos.

```
projetos/1-classificacao-mnist/
├── train_model.py         # ✏️ Treinamento do modelo
├── optimize_model.py      # ✏️ Conversão e otimização
├── run_inference.py       # ✏️ Inferência de exemplo com o modelo otimizado
├── requirements.txt       # 📄 Dependências do projeto
├── model.h5               # 🤖 Gerado por você — deve ser commitado
├── model.tflite           # ⚡ Gerado por você — deve ser commitado
└── README.md               # 📝 Este arquivo (também usado como relatório)
```

## ⚠️ Restrições e Considerações de Engenharia

- Entrada do modelo: imagens 28x28, 1 canal (grayscale), normalizadas em [0, 1]
- CNN simples — evite arquiteturas muito profundas
- Não utilize modelos pré-treinados
- Número de épocas limitado (ex: até 15, com early stopping)
- Treinamento apenas em CPU

## ⚖️ Critérios de Avaliação

- **Funcionalidade** — execução correta dos scripts e geração dos arquivos `.h5` e `.tflite`
- **Qualidade do modelo** — acurácia de validação consistente com o esperado para o dataset
- **Edge AI** — conversão correta para `.tflite` com técnica de otimização aplicada
- **Documentação** — preenchimento adequado do relatório abaixo

---

## 📝 Relatório do Candidato

👤 **Nome Completo:** Ronielly Santana

### 1️⃣ Resumo da Arquitetura do Modelo

Foi desenvolvida uma Rede Neural Convolucional (CNN) com entrada no formato 28x28 e um canal de cor, correspondente às imagens em escala de cinza do MNIST.

A rede possui três blocos convolucionais. Cada bloco é composto por uma camada Conv2D, uma camada BatchNormalization e uma camada MaxPooling2D. Foram utilizados, respectivamente, 32, 64 e 128 filtros convolucionais. Após os blocos, os dados são processados por uma camada Flatten, uma camada Dense com 64 neurônios e uma camada Dropout com taxa de 0,4. A saída utiliza 10 neurônios e ativação softmax, correspondentes aos dígitos de 0 a 9.

As imagens foram normalizadas para o intervalo entre 0 e 1. O conjunto original de treinamento foi embaralhado com semente fixa e dividido em 55.000 amostras de treino e 5.000 amostras de validação.

O treinamento foi realizado em CPU, por até 15 épocas, utilizando o otimizador Adam. Foi aplicado EarlyStopping com monitoramento da perda de validação, paciência de três épocas e restauração dos melhores pesos.

### 2️⃣ Bibliotecas Utilizadas

As principais bibliotecas utilizadas foram:

- Python 3.11;
- TensorFlow 2.20.0;
- Keras 3.11.3;
- NumPy 2.2.6.

### 3️⃣ Técnica de Otimização do Modelo

Foi utilizada a técnica Dynamic Range Quantization durante a conversão do modelo Keras para TensorFlow Lite. A otimização foi configurada com `tf.lite.Optimize.DEFAULT`.

Essa técnica quantiza principalmente os pesos do modelo, reduzindo o tamanho do arquivo e o consumo de memória, sem exigir um conjunto representativo para calibração. O objetivo é facilitar a execução do modelo em dispositivos de borda com recursos computacionais limitados.

### 4️⃣ Resultados Obtidos

A acurácia final no conjunto de validação foi de 98,84%.

Os tamanhos obtidos foram:

- `model.h5`: 1,99 MB;
- `model.tflite`: 0,17 MB;
- redução de tamanho: 91,24%.

O modelo otimizado preservou o funcionamento correto e apresentou uma redução expressiva de armazenamento, tornando-se mais adequado para aplicações de Edge AI.

### 5️⃣ Comentários Adicionais (Opcional)

O projeto permitiu aplicar o fluxo completo de uma solução de visão computacional: preparação dos dados, construção da CNN, treinamento, validação, salvamento, conversão, quantização e inferência com o modelo otimizado.

O uso de uma semente fixa tornou a preparação dos dados reproduzível. O treinamento foi realizado somente em CPU, conforme as restrições do desafio. A principal decisão de engenharia foi utilizar uma CNN relativamente compacta, capaz de alcançar boa acurácia sem criar um modelo excessivamente pesado.

### 6️⃣ Exemplo de Inferência

Saída obtida ao executar `run_inference.py` com o arquivo `model.tflite`:

```text
Amostra 1: predito=7 | real=7
Amostra 2: predito=2 | real=2
Amostra 3: predito=1 | real=1
Amostra 4: predito=0 | real=0
Amostra 5: predito=4 | real=4
```

O modelo otimizado classificou corretamente todas as cinco amostras analisadas. Esse resultado indica que a quantização reduziu significativamente o tamanho do modelo sem comprometer as inferências observadas nesse teste.