import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def tratar_dados(df):
    df_limpo = df.copy()
    df_limpo['stalk-root'] = df_limpo['stalk-root'].replace('?', 'missing')
    if 'veil-type' in df_limpo.columns:
        df_limpo = df_limpo.drop(columns=['veil-type'])
    return df_limpo

def preparar_dados(df):
    X = df.drop(columns=['class'])
    y = df['class']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_encoded = pd.get_dummies(X, drop_first=False)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42
    )
    return X_train, X_test, y_train, y_test, le

def treinar_e_avaliar(X_train, X_test, y_train, y_test):
    modelos = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5)
    }
    
    X_encoded = pd.concat([X_train, X_test])
    y_encoded = np.concatenate([y_train, y_test])
    
    resultados = {}
    for nome, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred, target_names=["Comestível (e)", "Venenoso (p)"], output_dict=True)
        cv_score = cross_val_score(modelo, X_encoded, y_encoded, cv=5).mean()
        
        resultados[nome] = {
            "modelo": modelo,
            "acuracia": acc,
            "acuracia_cv": cv_score,
            "matriz_confusao": cm,
            "relatorio_classificacao": cr
        }
    return resultados

def plotar_matriz_confusao(cm, classes, nome_modelo):
    labels_formatados = []
    for c in classes:
        if c == 'e' or c == "Comestível (e)":
            labels_formatados.append("e (Comestível)")
        elif c == 'p' or c == "Venenoso (p)":
            labels_formatados.append("p (Venenoso)")
        else:
            labels_formatados.append(str(c))
            
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels_formatados, yticklabels=labels_formatados, cbar=False, ax=ax)
    ax.set_title(f"Matriz de Confusão - {nome_modelo}", fontsize=11)
    ax.set_ylabel("Classe Real")
    ax.set_xlabel("Classe Predita")
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    from eda import carregar_dados
    
    os.makedirs("plots", exist_ok=True)
    df_original = carregar_dados()
    df_limpo = tratar_dados(df_original)
    
    X_train, X_test, y_train, y_test, le = preparar_dados(df_limpo)
    resultados = treinar_e_avaliar(X_train, X_test, y_train, y_test)
    
    for nome, res in resultados.items():
        print(f"\n==================== {nome.upper()} ====================")
        print(f"Acurácia Teste (Holdout): {res['acuracia'] * 100:.2f}%")
        print(f"Acurácia Validação Cruzada (CV): {res['acuracia_cv'] * 100:.2f}%")
        
        cr_texto = classification_report(y_test, res['modelo'].predict(X_test), target_names=["Comestível (e)", "Venenoso (p)"])
        print("\nClassification Report:")
        print(cr_texto)
        
        fig_cm = plotar_matriz_confusao(res['matriz_confusao'], le.classes_, nome)
        nome_arquivo = f"matriz_confusao_{nome.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
        fig_cm.savefig(os.path.join("plots", nome_arquivo), dpi=150)
        plt.close()
        
    print("\n==================== COMPARATIVO FINAL ====================")
    dados_comparativo = [[nome, res['acuracia'] * 100, res['acuracia_cv'] * 100] for nome, res in resultados.items()]
    df_comparacao = pd.DataFrame(dados_comparativo, columns=['Modelo', 'Acurácia Teste (%)', 'Acurácia CV (%)'])
    print(df_comparacao.sort_values(by='Acurácia CV (%)', ascending=False).to_string(index=False))
