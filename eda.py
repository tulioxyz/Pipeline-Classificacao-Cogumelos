import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")

def carregar_dados(caminho_csv="data/cogumelos.csv"):
    return pd.read_csv(caminho_csv)

def plotar_distribuicao_alvo(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x='class', data=df, order=['e', 'p'], hue='class', hue_order=['e', 'p'], legend=False, ax=ax)
    ax.set_title("Distribuição da Variável Alvo (class)", fontsize=12, pad=10)
    ax.set_xticklabels(["Comestível (e)", "Venenoso (p)"])
    ax.set_xlabel("Classe")
    ax.set_ylabel("Quantidade")
    
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=9)
    plt.tight_layout()
    return fig

def plotar_relacao_odor(df):
    df_plot = df.copy()
    odor_map = {
        'a': 'a (amêndoa)',
        'l': 'l (anis)',
        'c': 'c (creosoto)',
        'y': 'y (peixe)',
        'f': 'f (fétido)',
        'm': 'm (mofo)',
        'n': 'n (nenhum)',
        'p': 'p (pungente)',
        's': 's (picante)'
    }
    df_plot['odor'] = df_plot['odor'].map(odor_map)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(x='odor', hue='class', hue_order=['e', 'p'], data=df_plot, ax=ax)
    ax.set_title("Relação entre o Odor e a Comestibilidade", fontsize=12, pad=10)
    ax.set_xlabel("Odor")
    ax.set_ylabel("Quantidade")
    ax.legend(title="Classe", labels=["Comestível (e)", "Venenoso (p)"])
    
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=8)
                        
    plt.tight_layout()
    return fig

def plotar_relacao_esporos(df):
    df_plot = df.copy()
    spore_map = {
        'k': 'k (preto)',
        'n': 'n (marrom)',
        'b': 'b (amarelo-claro)',
        'h': 'h (chocolate)',
        'r': 'r (verde)',
        'o': 'o (laranja)',
        'u': 'u (roxo)',
        'w': 'w (branco)',
        'y': 'y (amarelo)'
    }
    df_plot['spore-print-color'] = df_plot['spore-print-color'].map(spore_map)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(x='spore-print-color', hue='class', hue_order=['e', 'p'], data=df_plot, ax=ax)
    ax.set_title("Relação entre a Cor dos Esporos e a Comestibilidade", fontsize=12, pad=10)
    ax.set_xlabel("Cor do Esporo")
    ax.set_ylabel("Quantidade")
    ax.legend(title="Classe", labels=["Comestível (e)", "Venenoso (p)"])
    
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=8)
                        
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)
    dados = carregar_dados()
    
    fig_alvo = plotar_distribuicao_alvo(dados)
    fig_alvo.savefig("plots/distribuicao_target.png", dpi=150)
    
    fig_odor = plotar_relacao_odor(dados)
    fig_odor.savefig("plots/relacao_odor.png", dpi=150)
    
    fig_esporos = plotar_relacao_esporos(dados)
    fig_esporos.savefig("plots/relacao_esporos.png", dpi=150)
    
    print("Gráficos de Análise Exploratória salvos na pasta 'plots/'!")
