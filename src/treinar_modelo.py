# aqui a gente treina o modelo de regressao pra prever a nota final
# e calcula as metricas pedidas: media do erro, desvio padrao e o intervalo de erro
# tambem gera os graficos pra colocar no relatorio

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    # se ja tiver o csv gerado a gente usa ele, senao gera na hora
    caminho_csv = "data/alunos.csv"
    if os.path.exists(caminho_csv):
        df = pd.read_csv(caminho_csv)
    else:
        df = gerar_base()
        os.makedirs("data", exist_ok=True)
        df.to_csv(caminho_csv, index=False)

    X = df[FEATURES]
    y = df[TARGET]

    # separa treino e teste, 75/25 como o professor pediu
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED
    )

    return df, X_train, X_test, y_train, y_test


def avaliar_modelo(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)

    # erro de cada previsao (real menos previsto)
    erros = y_test.values - y_pred

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    media_erro = erros.mean()
    desvio_erro = erros.std(ddof=1)

    # intervalo de erro de 95%, usando a formula da normal (1.96 * desvio padrao)
    # vi isso na aula de estatistica, da pra usar pq os erros parecem seguir uma normal
    ic_95 = 1.96 * desvio_erro

    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2: {r2:.3f}")
    print(f"media do erro: {media_erro:.3f}")
    print(f"desvio padrao do erro: {desvio_erro:.3f}")
    print(f"intervalo de erro (95%): +/- {ic_95:.3f} pontos")

    return {
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

    # grafico 1: nota real vs nota que o modelo previu
    # quanto mais perto da linha pontilhada, melhor a previsao
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color="#2563eb", edgecolor="white")
    plt.plot([0, 10], [0, 10], "--", color="gray", label="previsao perfeita")
    plt.xlabel("nota final real")
    plt.ylabel("nota final prevista")
    plt.title("nota real vs nota prevista")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(GRAF_DIR, "real_vs_previsto.png"), dpi=150)
    plt.close()

    # grafico 2: histograma dos erros, pra ver se ta distribuido tipo uma normal
    plt.figure(figsize=(6, 4))
    plt.hist(erros, bins=20, color="#2563eb", edgecolor="white")
    plt.axvline(resultado["media_erro"], color="red", linestyle="--", label="media do erro")
    plt.xlabel("erro (real - previsto)")
    plt.ylabel("frequencia")
    plt.title("distribuicao dos erros de previsao")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(GRAF_DIR, "distribuicao_erros.png"), dpi=150)
    plt.close()

    # grafico 3: correlacao entre as variaveis, so pra entender quais pesam mais
    plt.figure(figsize=(6, 5))
    corr = df[FEATURES + [TARGET]].corr()
    im = plt.imshow(corr, cmap="Blues", vmin=-1, vmax=1)
    plt.colorbar(im, label="correlacao")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.columns)), corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.title("matriz de correlacao")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAF_DIR, "matriz_correlacao.png"), dpi=150)
    plt.close()

    print(f"graficos salvos em {GRAF_DIR}/")


def main():
    df, X_train, X_test, y_train, y_test = preparar_dados()

    # usando regressao linear, que foi o modelo que fez mais sentido pro problema
    # (a nota final tem uma relacao mais ou menos linear com as variaveis)
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    resultado = avaliar_modelo(modelo, X_test, y_test)

    gerar_graficos(df, y_test, resultado)

    # salva um resumo em txt pra usar no relatorio depois
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "resumo_resultados.txt"), "w", encoding="utf-8") as f:
        f.write("=== Regressao Linear ===\n")
        f.write(f"MAE: {resultado['mae']:.3f}\n")
        f.write(f"RMSE: {resultado['rmse']:.3f}\n")
        f.write(f"R2: {resultado['r2']:.3f}\n")
        f.write(f"Media do erro: {resultado['media_erro']:.3f}\n")
        f.write(f"Desvio padrao do erro: {resultado['desvio_erro']:.3f}\n")
        f.write(f"Intervalo de erro (95%): +/-{resultado['ic_95']:.3f}\n")

    print(f"resumo salvo em {OUT_DIR}/resumo_resultados.txt")


if __name__ == "__main__":
    main()
