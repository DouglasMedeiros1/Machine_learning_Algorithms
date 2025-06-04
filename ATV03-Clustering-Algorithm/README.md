# Atividade 03 – Algoritmo de Agrupamento de Jogos

## Visão Geral
Este projeto, denominado **Atividade 03**, tem como objetivo desenvolver um algoritmo de clustering (agrupamento) usando um dataset real de jogos da Steam. O modelo agrupa jogos com base em características como gêneros, tags e compatibilidade com Linux.

## Dataset Utilizado
- Arquivo: `steam.csv`
- Origem: Dados públicos da Steam contendo informações de jogos, incluindo:
  - `appid`: identificador único do jogo
  - `name`: nome do jogo
  - `genres`: gêneros do jogo (por exemplo, "Action;Adventure")
  - `steamspy_tags`: tags atribuídas ao jogo separadas por vírgulas
  - `platforms`: plataformas suportadas (por exemplo, "windows;mac;linux")
- Observação: O dataset é carregado em `main.py` a partir do caminho `ATV03/steam.csv`.

## Estrutura de Arquivos
- **main.py**: Script principal que realiza todo o fluxo do projeto de clustering.
- **steam.csv**: Arquivo CSV com os dados limpos de jogos da Steam.
- **README_ATV03.md**: Este arquivo de descrição do projeto.

## Dependências
Para executar o script, é necessário ter instalado:
- Python 3.6 ou superior
- Bibliotecas Python:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `scikit-learn`

Instalação:
```bash
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
pip install pandas numpy matplotlib scikit-learn
```

## Descrição do Script (main.py)
1. **Carregamento dos Dados**
   - O script lê o arquivo `steam.csv` para um DataFrame do pandas.
   - Remove duplicatas com base na coluna `appid`.
   - Preenche valores ausentes em `genres`, `steamspy_tags` e `platforms` com strings vazias.

2. **One-Hot Encoding de Gêneros**
   - Divide a coluna `genres` (separada por `;`) em colunas binárias para cada gênero presente.

3. **One-Hot Encoding de Tags**
   - Extrai todas as tags em `steamspy_tags`, identifica as 20 mais frequentes e cria colunas binárias indicando presença de cada uma.

4. **Compatibilidade com Linux**
   - Cria a coluna `linux_binary` como 1 se o jogo tem suporte a Linux (verificando se `platforms` contém "linux"), caso contrário 0.

5. **Combinação de Features**
   - Concatena as colunas de gêneros, tags e `linux_binary` em uma matriz de features para clustering.

6. **Cálculo de WCSS (Within-Cluster Sum of Squares)**
   - Itera valores de \( k \) de 2 até 15.
   - Para cada \( k \), treina um modelo K-Means e registra o WCSS (inércia).

7. **Método do Cotovelo Automático**
   - Usa a função `optimal_number_of_clusters` para calcular o ponto de “cotovelo” automaticamente a partir dos valores de WCSS.
   - Determina o \( k \) ótimo que maximiza a distância ao segmento formado pelos WCSS em \( k=2 \) e \( k=	ext{máximo testado}\).

8. **Plot da Curva de Cotovelo**
   - Gera um gráfico da curva WCSS vs. \( k \), destacando o \( k \) ótimo com uma linha vertical vermelha.

9. **Aplicação do K-Means com \( k \) Ótimo**
   - Treina o modelo K-Means usando o \( k \) determinado.
   - Atribui cada jogo a um dos clusters e salva o rótulo no DataFrame (`df['cluster']`).

10. **Inspeção dos Centrôides**
    - Calcula a média das features binárias para cada centróide e exibe um DataFrame que mostra o perfil médio de cada cluster.

11. **Proporção de Suporte a Linux por Cluster**
    - Agrupa por `cluster` e calcula a proporção de jogos com `linux_binary = 1` em cada grupo.

12. **Visualização com PCA 2D**
    - Aplica PCA para reduzir as features a 2 componentes principais.
    - Plota cada jogo em um gráfico de dispersão colorido conforme seu cluster e exibe os centróides projetados.

13. **Visualização com PCA 3D**
    - Aplica PCA para reduzir as features a 3 componentes principais.
    - Plota em um gráfico 3D a dispersão dos jogos e posiciona os centróides.

14. **Análise de Compatibilidade Linux por Gênero**
    - Concatena `df_genres` e `linux_binary` para calcular a proporção de suporte a Linux para cada gênero.
    - Exibe um DataFrame com a proporção de cada gênero e imprime os top 10 gêneros com menor e maior suporte a Linux.
    - Imprime um “pódio” dos 3 gêneros com menor compatibilidade e dos 3 gêneros com maior compatibilidade a Linux.

## Como Executar
1. Navegue até a pasta do projeto no terminal.
2. Ative o ambiente virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # No Windows: venv\Scripts\activate
   ```
3. Instale as dependências:
   ```bash
   pip install pandas numpy matplotlib scikit-learn
   ```
4. Execute o script:
   ```bash
   python main.py
   ```
5. O script exibirá no console as etapas de pré-processamento, cálculo do método do cotovelo, perfis de centróides e proporções de compatibilidade a Linux. Também abrirá gráficos interativos de PCA 2D, PCA 3D e da curva de cotovelo.

## Resultados Esperados
- Gráfico do método do cotovelo indicando o \( k \) ótimo para clustering.
- Perfis médios dos centróides mostrando a distribuição binária de gêneros e tags por cluster.
- Proporção de suporte a Linux em cada cluster.
- Plots de PCA 2D e PCA 3D para visualizar a separação dos clusters.
- Tabela com proporção de compatibilidade a Linux para cada gênero, listando os top 10 gêneros com menor e maior suporte.
- Impressão de um “pódio” dos três gêneros com pior e melhor compatibilidade a Linux.

## Considerações Finais
Este projeto demonstra um pipeline completo de clustering:
1. Pré-processamento de dados categóricos (gêneros e tags) e criação de features.
2. Aplicação do método do cotovelo para determinar \( k \) automaticamente.
3. Treinamento e análise de K-Means.
4. Visualizações em PCA 2D e 3D para validar a separação dos grupos.
5. Insights sobre compatibilidade a Linux por gênero.


---
