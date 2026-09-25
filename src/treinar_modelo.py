"""
Treina um modelo de regressão para prever a nota final dos alunos e avalia
seu desempenho com métricas estatísticas: média e desvio padrão dos erros,
e o intervalo de erro (IC 95%) das previsões.

Gera também os gráficos de análise em outputs/graficos/.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from gerar_dados import gerar_base

SEED = 42
FEATURES = ["horas_estudo", "frequencia", "atividades_entregues", "nota_anterior"]
TARGET = "nota_final"

OUT_DIR = "outputs"
GRAF_DIR = os.path.join(OUT_DIR, "graficos")


def preparar_dados():
    caminho_csv = "data/alunos.csv"
    if os.path.exists(caminho_csv):
        df = pd.read_csv(caminho_csv)
    else:
        df = gerar_base()
        os.makedirs("data", exist_ok=True)
        df.to_csv(caminho_csv, index=False)

    X = df[FEATURES]
    y = df[TARGET]
    return df, train_test_split(X, y, test_size=0.25, random_state=SEED)


def avaliar_modelo(nome, modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    erros = y_test.values - y_pred  # erro = valor real - valor previsto

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    media_erro = erros.mean()
    desvio_erro = erros.std(ddof=1)

    # Intervalo de erro (95%) assumindo distribuição aproximadamente normal dos erros
    ic_95 = 1.96 * desvio_erro

    print(f"\n=== {nome} ===")
    print(f"MAE  (erro absoluto médio): {mae:.3f}")
    print(f"RMSE (raiz do erro quadrático médio): {rmse:.3f}")
    print(f"R²   (coeficiente de determinação): {r2:.3f}")
    print(f"Média do erro (viés): {media_erro:.3f}")
    print(f"Desvio padrão do erro: {desvio_erro:.3f}")
    print(f"Intervalo de erro (95%): ±{ic_95:.3f} pontos")

    return {
        "nome": nome,
        "y_pred": y_pred,
        "erros": erros,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "media_erro": media_erro,
        "desvio_erro": desvio_erro,
        "ic_95": ic_95,
    }


def gerar_graficos(df, y_test, resultado):
    os.makedirs(GRAF_DIR, exist_ok=True)
    y_pred = resultado["y_pred"]
    erros = resultado["erros"]

    # 1) Dispersão: nota real vs. nota prevista
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color="#2563eb", edgecolor="white")
    lim = [0, 10]
    plt.plot(lim, lim, "--", color="gray", label="Previsão perfeita")
    plt.xlabel("Nota final real")
    plt.ylabel("Nota final prevista")
    plt.title("Nota real vs. Nota prevista")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(GRAF_DIR, "real_vs_previsto.png"), dpi=150)
    plt.close()

    # 2) Histograma dos erros de previsão
    plt.figure(figsize=(6, 4))
    plt.hist(erros, bins=20, color="#2563eb", edgecolor="white")
    plt.axvline(resultado["media_erro"], color="red", linestyle="--", label="Média do erro")
    plt.xlabel("Erro (real - previsto)")
    plt.ylabel("Frequência")
    plt.title("Distribuição dos erros de previsão")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(GRAF_DIR, "distribuicao_erros.png"), dpi=150)
    plt.close()

    # 3) Importância das variáveis (Random Forest)
    if hasattr(resultado.get("modelo"), "feature_importances_"):
        importancias = resultado["modelo"].feature_importances_
        plt.figure(figsize=(6, 4))
        plt.barh(FEATURES, importancias, color="#2563eb")
        plt.xlabel("Importância relativa")
        plt.title("Importância das variáveis no modelo")
        plt.tight_layout()
        plt.savefig(os.path.join(GRAF_DIR, "importancia_variaveis.png"), dpi=150)
        plt.close()

    # 4) Correlação entre variáveis
    plt.figure(figsize=(6, 5))
    corr = df[FEATURES + [TARGET]].corr()
    im = plt.imshow(corr, cmap="Blues", vmin=-1, vmax=1)
    plt.colorbar(im, label="Correlação")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.columns)), corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.title("Matriz de correlação")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAF_DIR, "matriz_correlacao.png"), dpi=150)
    plt.close()

    print(f"\nGráficos salvos em {GRAF_DIR}/")


def main():
    df, (X_train, X_test, y_train, y_test) = preparar_dados()

    # Modelo 1: Regressão Linear (baseline simples e interpretável)
    modelo_linear = LinearRegression()
    modelo_linear.fit(X_train, y_train)
    resultado_linear = avaliar_modelo("Regressão Linear", modelo_linear, X_test, y_test)
    resultado_linear["modelo"] = modelo_linear

    # Modelo 2: Random Forest (captura relações não lineares)
    modelo_rf = RandomForestRegressor(n_estimators=200, random_state=SEED)
    modelo_rf.fit(X_train, y_train)
    resultado_rf = avaliar_modelo("Random Forest", modelo_rf, X_test, y_test)
    resultado_rf["modelo"] = modelo_rf

    # Usa o melhor modelo (maior R²) para os gráficos finais
    melhor = resultado_rf if resultado_rf["r2"] >= resultado_linear["r2"] else resultado_linear
    print(f"\n>> Melhor modelo: {melhor['nome']} (R² = {melhor['r2']:.3f})")

    gerar_graficos(df, y_test, melhor)

    # Salva um resumo em texto para consulta rápida / apoio ao relatório
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "resumo_resultados.txt"), "w", encoding="utf-8") as f:
        for r in (resultado_linear, resultado_rf):
            f.write(f"=== {r['nome']} ===\n")
            f.write(f"MAE: {r['mae']:.3f}\n")
            f.write(f"RMSE: {r['rmse']:.3f}\n")
            f.write(f"R2: {r['r2']:.3f}\n")
            f.write(f"Media do erro: {r['media_erro']:.3f}\n")
            f.write(f"Desvio padrao do erro: {r['desvio_erro']:.3f}\n")
            f.write(f"Intervalo de erro (95%): +/-{r['ic_95']:.3f}\n\n")
        f.write(f"Melhor modelo: {melhor['nome']} (R2 = {melhor['r2']:.3f})\n")

    print(f"Resumo salvo em {OUT_DIR}/resumo_resultados.txt")


if __name__ == "__main__":
    main()
