# Atividade 01 – Algoritmo de Classificação de Comentários

## Visão Geral
Este projeto, denominado **Atividade 01**, tem como objetivo desenvolver um algoritmo de classificação de comentários baseado em um dataset real. O modelo classifica cada comentário em uma das categorias:
- **crítica negativa**: comentários que não acrescentam ou não desenvolvem o debate.
- **crítica construtiva**: comentários que criticam uma decisão ou situação, mas agregam ao debate.
- **proposta**: comentários que apresentam possíveis soluções ou sugestões para o contexto em discussão.

## Dataset Utilizado
- Arquivo: `War_Data_Sentiment.csv`
- Fonte: [Kaggle – The People Opinions About The War In 2023](https://www.kaggle.com/datasets/aravindhmp/the-people-opinions-about-the-war-in-2023)
- Descrição: Contém comentários de usuários sobre a guerra em 2023, juntamente com métricas de sentimento (pontuação composta) e categoria de emoção (positivo, negativo, neutro).

## Estrutura de Arquivos
- **ATV01-MachineLearning-Douglas.py**: Script principal que realiza todo o fluxo do projeto.
- **War_Data_Sentiment.csv**: Arquivo CSV com os dados que serão carregados e processados.
- **README_ATV01.md**: Este arquivo de descrição do projeto.

## Dependências
Para executar o script, é necessário ter instalado:
- Python 3.6 ou superior
- Bibliotecas Python:
  - `pandas`
  - `numpy`
  - `scikit-learn`

Instalação:
```bash
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
pip install pandas numpy scikit-learn
```

## Descrição do Script (ATV01-MachineLearning-Douglas.py)
1. **Carregamento dos Dados**  
   - O script carrega o arquivo `War_Data_Sentiment.csv` em um DataFrame do pandas.  
   - Remove colunas desnecessárias (como `id`) e linhas com valores nulos.

2. **Mapeamento de Categorias**  
   - Define uma função `map_category` que classifica cada comentário em três categorias personalizadas.  
   - Utiliza palavras-chave (ex.: "should", "need to", "i think", "i believe") e métricas de sentimento (`Compound`) para determinar a categoria:
     - Se o comentário contiver termos de sugestão, mapeia para **proposta**.
     - Se contiver termos de opinião ou crítica construtiva, mapeia para **crítica construtiva**.
     - Se o score composto (`Compound`) for menor que -0.3 ou a emoção for negativa, mapeia para **crítica negativa**.
     - Emoções positivas são mapeadas como **crítica construtiva** por padrão.
   - Cria a coluna `MappedCategory` no DataFrame com os rótulos resultantes.

3. **Preparação dos Dados para Modelagem**  
   - Separa as variáveis independentes (`X = comments`) e a variável dependente (`y = MappedCategory`).
   - Aplica `TfidfVectorizer` para converter o texto dos comentários em vetores TF-IDF (máximo de 5000 features e remoção de stopwords em inglês).

4. **Divisão em Conjuntos de Treino e Teste**  
   - Utiliza `train_test_split` para dividir os dados em treino (80%) e teste (20%), garantindo estratificação pela variável alvo (`stratify=y`).

5. **Treinamento do Modelo**  
   - Treina um classificador **Random Forest** com os seguintes parâmetros:
     - `n_estimators=300` (número de árvores)
     - `min_samples_split=10`
     - `min_samples_leaf=5`
     - `random_state=42`

6. **Avaliação do Modelo**  
   - Realiza previsões no conjunto de teste.
   - Calcula métricas de avaliação:
     - **Acurácia**
     - **Precisão (Precision)**
     - **Recall**
     - **F1-Score**
   - Exibe o relatório de classificação completo (`classification_report`) mostrando pontuações por classe.

## Como Executar
1. Abra o terminal na pasta do projeto.
2. Ative o ambiente virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # No Windows: venv\Scripts\activate
   ```
3. Instale as dependências:
   ```bash
   pip install pandas numpy scikit-learn
   ```
4. Execute o script:
   ```bash
   python ATV01-MachineLearning-Douglas.py
   ```

O script exibirá no console:
- Informações sobre o carregamento e limpeza dos dados.
- Exemplo dos comentários e categorias mapeadas.
- Progresso de vetorização, divisão, treinamento e avaliação.
- Métricas finais de desempenho do modelo no conjunto de teste.

## Resultados Esperados
- Impressão das métricas de avaliação com valores de acurácia, precisão, recall e F1-Score.
- Relatório de classificação detalhado para cada uma das três categorias: crítica negativa, crítica construtiva e proposta.

## Considerações Finais
Este projeto demonstra um fluxo básico de processamento de texto e classificação de comentários:
1. Carregamento e limpeza de dados.
2. Mapeamento de categorias personalizadas.
3. Vetorização de texto com TF-IDF.
4. Treinamento e avaliação de um modelo de Random Forest.

---