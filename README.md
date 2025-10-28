# Controle de Qualidade Visual com CNN e Transfer Learning

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-blueviolet?logo=scikit-learn)

## 1. Visão Geral do Projeto

Este é um projeto **end-to-end** de **Visão Computacional** aplicado ao **Controle de Qualidade** industrial. O objetivo é classificar automaticamente imagens de peças metálicas fundidas (casting) como "Aprovadas" ou "Defeituosas" com base em defeitos visuais.

Utilizando o dataset [Casting Product Image Data for Quality Control](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product), uma **Rede Neural Convolucional (CNN)** pré-treinada (`MobileNetV2`) é adaptada através de **Transfer Learning** para esta tarefa específica. O resultado final é um **Dashboard Interativo (Streamlit)** onde o usuário pode fazer o upload de uma imagem de peça e receber um veredito instantâneo sobre sua qualidade.

## 2. O Problema de Negócio: Automatizando a Inspeção Visual

A inspeção visual manual é um gargalo comum em muitas linhas de produção. Ela é:
* **Lenta:** Limita a velocidade da produção.
* **Subjetiva:** A interpretação de "defeito" pode variar entre inspetores.
* **Cara:** Requer mão de obra dedicada.
* **Propensa a Erros:** Fadiga e falta de atenção podem levar a classificações incorretas.

A Visão Computacional oferece uma solução para automatizar essa tarefa, trazendo:
* **Velocidade:** Classificação em milissegundos.
* **Consistência:** Critérios objetivos aplicados uniformemente.
* **Eficiência:** Libera os operadores humanos para tarefas mais complexas.
* **Precisão:** Modelos bem treinados podem superar a precisão humana em tarefas repetitivas.

Este projeto aborda diretamente a necessidade de um sistema automatizado e confiável para o controle de qualidade visual.

## 3. A Solução de Machine Learning: Transfer Learning e CNNs

Treinar uma CNN do zero para tarefas de visão requer datasets massivos e muito poder computacional. A abordagem utilizada aqui é o **Transfer Learning**, que aproveita o conhecimento de modelos pré-treinados:

1.  **Modelo Base:** Utilizamos o `MobileNetV2`, uma CNN leve e eficiente pré-treinada pela Google no gigantesco dataset ImageNet (milhões de imagens de objetos diversos). Este modelo já é um "especialista" em reconhecer padrões visuais como bordas, texturas e formas.
2.  **Congelamento:** As camadas convolucionais do `MobileNetV2` foram "congeladas" (`trainable=False`), preservando seu conhecimento genérico de visão.
3.  **Nova "Cabeça":** Uma pequena rede neural (camadas `GlobalAveragePooling2D`, `Dropout`, `Dense`) foi adicionada ao topo do modelo base congelado.
4.  **Treinamento Focado:** Apenas essa nova "cabeça" foi treinada com as imagens específicas do nosso dataset de peças fundidas. O modelo aprendeu rapidamente a *combinar* os recursos visuais genéricos detectados pelo `MobileNetV2` para classificar as peças como "ok" ou "defeituosas".
5.  **Data Augmentation:** Para aumentar a robustez e evitar overfitting, transformações aleatórias (rotação, zoom, flip horizontal) foram aplicadas às imagens de treino durante o processo, ensinando o modelo a generalizar melhor.
6.  **Pré-processamento:** As imagens foram redimensionadas para `(224, 224)` e os valores dos pixels normalizados para o intervalo `[-1, 1]`, conforme esperado pelo `MobileNetV2`. Essa etapa foi integrada ao pipeline de dados `tf.data`.

## 4. O Dashboard Interativo (Streamlit)

O resultado final é uma aplicação web simples e intuitiva construída com `streamlit`. O dashboard permite:
* **Upload Fácil:** O usuário seleciona um arquivo de imagem (`.jpg`, `.jpeg`, `.png`) do seu computador.
* **Pré-processamento Automático:** A imagem carregada é redimensionada, convertida para RGB (se necessário) e normalizada nos bastidores.
* **Previsão Rápida:** A imagem processada é enviada ao modelo Keras carregado para obter a classificação.
* **Veredito Claro:** O resultado é exibido de forma destacada: "✅ APROVADO" ou "❌ DEFEITUOSO", junto com a porcentagem de confiança do modelo.

*(Exemplo do Dashboard em ação)*
`![Dashboard de Controle de Qualidade](dashboard_qc_demo.png)`

## 5. Tech Stack (Tecnologias Utilizadas)

* **Dashboard:** Streamlit
* **Deep Learning / Visão Computacional:** TensorFlow (Keras)
* **Manipulação de Dados:** NumPy
* **Processamento de Imagem:** Pillow (PIL)
* **Estrutura de Arquivos:** Pathlib

## 6. Estrutura do Projeto

```
[ projeto_controle_qualidade/ ]
|
|-- data/
|   |-- casting_data/      <-- (Dataset descompactado)
|       |-- train/
|           |-- ok_front/
|           |-- def_front/
|       |-- test/
|           |-- ok_front/
|           |-- def_front/
|
|-- dashboard/
|   |-- app.py                  <-- (Script do Streamlit)
|   |-- requirements_dash.txt
|
|-- models/
|   |-- quality_control_model.h5 <-- (Modelo treinado)
|
|-- notebooks/
|   |-- 01_Treinamento_CNN.ipynb
|
|-- requirements.txt            <-- (Dependências do notebook)
|-- README.md
|-- dashboard_qc_demo.png       <-- (Screenshot do dashboard)
|-- confusion_matrix.png        <-- (Screenshot da matriz de confusão)
```

## 7. Como Usar

### Pré-requisitos
* Python 3.11+
* Ambiente virtual (recomendado)

### Instalação
1.  Clone o repositório e entre na pasta.
2.  Crie e ative um ambiente virtual.
3.  Instale as dependências de treinamento:
    ```bash
    pip install -r requirements.txt
    ```
4.  Instale as dependências do dashboard:
    ```bash
    pip install -r dashboard/requirements_dash.txt
    ```

### Passo 1: Treinar o Modelo (Opcional - Modelo já treinado no repo)
1.  Abra e execute o notebook `notebooks/01_Treinamento_CNN.ipynb`.
2.  Execute todas as células (Célula 1, 2 e 3).
3.  Isso salvará o arquivo `quality_control_model.h5` na pasta `models/`.

### Passo 2: Executar o Dashboard
1.  No seu terminal, a partir da pasta raiz do projeto, execute o Streamlit:
    ```bash
    streamlit run dashboard/app.py
    ```
2.  O Streamlit abrirá no seu navegador.
3.  Clique em "Browse files" e faça o upload de uma imagem da pasta `data/casting_data/test/ok_front/` ou `data/casting_data/test/def_front/`.
4.  Observe o veredito do modelo.

## 8. Resultados do Modelo

O modelo treinado alcançou uma **acurácia de 98.18%** no conjunto de teste (imagens nunca vistas durante o treinamento), demonstrando excelente capacidade de generalização.

O Relatório de Classificação e a Matriz de Confusão (gerados na Célula 3 do notebook) confirmam o alto desempenho:

*(Insira aqui o Classification Report da sua execução, se desejar)*

**Matriz de Confusão:**
`![Matriz de Confusão](confusion_matrix.png)`

A matriz mostra que o modelo cometeu muito poucos erros, classificando corretamente a vasta maioria das peças "ok" e "defeituosas". Isso indica que o sistema é confiável para uso em um cenário de controle de qualidade real.
