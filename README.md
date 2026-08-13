# Pipeline de Classificação de Cogumelos

Este projeto implementa um pipeline de dados e aprendizado de máquina para classificar cogumelos entre **comestíveis (e)** e **venenosos (p)** com base em suas características morfológicas e físicas, utilizando o clássico conjunto de dados da UCI (Mushroom Classification).

## Como Clonar e Executar o Projeto

Siga os passos abaixo para preparar o ambiente e rodar as análises localmente.

### 1. Pré-requisitos
Certifique-se de ter o Python 3 instalado em sua máquina.

### 2. Clonar o Repositório
Abra o terminal e execute o comando abaixo para clonar este repositório:
```bash
git clone https://github.com/tulioxyz/Pipeline-Classificacao-Cogumelos.git
cd Pipeline-Classificacao-Cogumelos
```

### 3. Configurar o Ambiente Virtual (Recomendado)
Para evitar conflitos de bibliotecas, é recomendável criar e ativar um ambiente virtual:

**No Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**No Windows (CMD / PowerShell):**
```cmd
python -m venv venv
venv\Scripts\activate
```

### 4. Instalar as Dependências
Instale as bibliotecas necessárias para executar o projeto:
```bash
pip install -r requirements.txt
```

---

## Executando as Etapas do Projeto

O projeto é composto por scripts em código puro Python e um Jupyter Notebook:

### A. Análise Exploratória de Dados (EDA)
Para executar a análise exploratória (carregar o dataset e gerar os gráficos de análise de dados na pasta `plots/`):
```bash
python eda.py
```

### B. Treinamento e Avaliação dos Modelos
Para treinar os modelos de Machine Learning (Decision Tree, Random Forest e KNN), gerar as tabelas comparativas e salvar as matrizes de confusão na pasta `plots/`:
```bash
python main.py
```

### C. Jupyter Notebook
Caso prefira executar e visualizar o passo a passo interativo no Jupyter Notebook:
1. Com o seu ambiente virtual ativado, instale o Jupyter (se ainda não o tiver):
   ```bash
   pip install jupyter
   ```
2. Inicie o servidor do Jupyter:
   ```bash
   jupyter notebook
   ```
3. Abra e execute o arquivo `cogumelos_classificacao.ipynb`.

---

## Estrutura do Repositório

```
Pipeline-Classificacao-Cogumelos/
├── data/
│   └── cogumelos.csv              # Conjunto de dados (local)
├── plots/                         # Gráficos e matrizes de confusão salvos pelos scripts
├── .gitignore                     # Arquivos ignorados pelo Git
├── cogumelos_classificacao.ipynb  # Notebook com a análise passo a passo
├── eda.py                         # Script de Análise Exploratória de Dados
├── main.py                        # Script de modelagem e treinamento
├── README.md                      # Instruções do projeto
└── requirements.txt               # Lista de dependências do projeto
```