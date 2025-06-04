# README Principal – Projeto de Machine Learning

Este repositório contém três atividades de Machine Learning desenvolvidas em Python, utilizando datasets reais para diferentes objetivos e tipos de algoritmos. A seguir, uma visão geral de cada atividade, seu objetivo, o algoritmo principal utilizado e o dataset correspondente.

---

## Atividade 01 – Algoritmo de Classificação de Comentários

- **Objetivo**: Criar um modelo que classifique comentários de usuários em três categorias:
  1. Crítica Negativa
  2. Crítica Construtiva
  3. Proposta
- **Algoritmo Principal**: Random Forest (classificação)
- **Fluxo Resumido**:
  1. Carregamento e limpeza do dataset.
  2. Mapeamento de categorias personalizadas a partir de métricas de sentimento e palavras-chave.
  3. Vetorização de texto usando TF-IDF.
  4. Treino e avaliação de um classificador Random Forest.
- **Dataset**: `War_Data_Sentiment.csv`  
  - Fonte: Kaggle – The People Opinions About The War In 2023  
  - Contém comentários de usuários sobre guerra com pontuação de sentimento e rótulo de emoção.
- **Arquivo Principal**: `ATV01-MachineLearning-Douglas.py`
- **Link para detalhes**: [README_ATV01.md](README_ATV01.md)

---

## Atividade 02 – Algoritmo de Regressão com Séries Temporais

- **Objetivo**: Prever o preço médio de jogos na Steam ao longo do tempo utilizando regressão de séries temporais.
- **Algoritmo Principal**: Facebook Prophet (modelagem de séries temporais)
- **Fluxo Resumido**:
  1. Leitura e pré-processamento do dataset, conversão de datas e agregação diária.
  2. Criação de regressores extras (razão de avaliações e log de proprietários).
  3. Treino do modelo Prophet no conjunto de treino e previsão para o conjunto de teste.
  4. Avaliação com métricas (MSE, RMSE, MAE, R²) e visualização dos componentes sazonais e de tendência.
- **Dataset**: `steam_store_games_clean.csv`  
  - Origem: Dados de jogos da Steam com informações de data de lançamento, preço, avaliações e número de proprietários.
- **Arquivo Principal**: `ATV02-Douglas.py`
- **Link para detalhes**: [README_ATV02.md](README_ATV02.md)

---

## Atividade 03 – Algoritmo de Agrupamento de Jogos

- **Objetivo**: Agrupar jogos da Steam em clusters com base em características como gêneros, tags e compatibilidade com Linux.
- **Algoritmo Principal**: K-Means (clustering)
- **Fluxo Resumido**:
  1. Carregamento e limpeza do dataset.
  2. One-Hot Encoding de gêneros e das 20 tags mais frequentes.
  3. Criação de indicador de compatibilidade com Linux.
  4. Cálculo do método do cotovelo para determinar o número ótimo de clusters (\(k\)).
  5. Treino do modelo K-Means e atribuição de clusters aos jogos.
  6. Análise de centróides, proporção de suporte a Linux por cluster e visualizações via PCA 2D e 3D.
- **Dataset**: `steam.csv`  
  - Origem: Dados públicos de jogos da Steam contendo `appid`, `name`, `genres`, `steamspy_tags` e `platforms`.
- **Arquivo Principal**: `main.py`
- **Link para detalhes**: [README_ATV03.md](README_ATV03.md)

---

## Como Navegar neste Repositório

1. **Leia este README Principal** para entender a estrutura geral e os objetivos de cada atividade.
2. **Abra o README de cada atividade** (links acima) para obter detalhes sobre instalação, execução, descrições completas dos scripts e resultados esperados.
3. **Execute os scripts** conforme indicado em cada README específico para reproduzir os experimentos e visualizações.

---

### Dependências Comuns

Para todas as atividades, recomenda-se criar um ambiente virtual Python e instalar bibliotecas essenciais:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib` (quando aplicável)
- `prophet` (somente Atividade 02)
```bash
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
pip install pandas numpy scikit-learn matplotlib prophet
```

---

### Estrutura de Arquivos

```
├── ATV01-MachineLearning-Douglas.py
├── README_ATV01.md
├── War_Data_Sentiment.csv
│
├── ATV02-Douglas.py
├── README_ATV02.md
├── steam_store_games_clean.csv
│
├── main.py
├── README_ATV03.md
├── steam.csv
│
└── README_PRINCIPAL.md   ← Este arquivo
```

---

## Contato

Para dúvidas ou sugestões sobre as atividades, entre em contato com o autor do repositório, eu.

---
