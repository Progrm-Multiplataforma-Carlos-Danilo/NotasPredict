"""
Treina um modelo de regressão para prever a nota final dos alunos e avalia
seu desempenho com métricas estatísticas: média e desvio padrão dos erros,
e o intervalo de erro (IC 95%) das previsões.

Os gráficos são gerados com matplotlib (backend Agg, em memória) e depois
renderizados como arte ASCII diretamente no terminal — sem salvar nenhum
arquivo de imagem em disco.
"""

import io
import os

import matplotlib

matplotlib.use("Agg")  # renderiza em memória, sem abrir janela nem salvar arquivo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from gerar_dados import gerar_base

SEED = 42
FEATURES = ["horas_estudo", "frequencia", "atividades_entregues", "nota_anterior"]
TARGET = "nota_final"

OUT_DIR = "outputs"

# Rampa de caracteres do mais escuro (denso) ao mais claro (vazio)
RAMPA_ASCII = "@%#*+=-:. "


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


def figura_para_ascii(fig, largura=100):
    """Renderiza uma figura matplotlib em memória (PNG) e converte para ASCII art."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110)
    plt.close(fig)
    buffer.seek(0)

    img = Image.open(buffer).convert("L")  # escala de cinza
    largura_original, altura_original = img.size
    # caracteres de terminal são ~2x mais altos que largos: compensa a proporção
    altura = int(largura * (altura_original / largura_original) * 0.5)
    img = img.resize((largura, max(1, altura)))

    pixels = np.array(img)
    indices = (pixels / 255 * (len(RAMPA_ASCII) - 1)).astype(int)

    linhas = ["".join(RAMPA_ASCII[i] for i in linha) for linha in indices]
    return "\n".join(linhas)


def grafico_real_vs_previsto(y_test, y_pred):
    print("\n--- Nota real vs. Nota prevista ---")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, alpha=0.7, color="black", edgecolor="black")
    lim = [0, 10]
    ax.plot(lim, lim, "--", color="black", linewidth=1)
    ax.set_xlabel("Nota real")
    ax.set_ylabel("Nota prevista")
    ax.set_title("Nota real vs. Nota prevista")
    fig.tight_layout()
    print(figura_para_ascii(fig))


def grafico_distribuicao_erros(erros, media_erro):
    print("\n--- Distribuição dos erros de previsão ---")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(erros, bins=20, color="black", edgecolor="white")
    ax.axvline(media_erro, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Erro (real - previsto)")
    ax.set_ylabel("Frequência")
    ax.set_title("Distribuição dos erros de previsão")
    fig.tight_layout()
    print(figura_para_ascii(fig))


def grafico_correlacao(df):
    print("\n--- Correlação de cada variável com a nota final ---")
    corr = df[FEATURES + [TARGET]].corr()[TARGET].drop(TARGET)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(corr.index, corr.values, color="black")
    ax.set_xlabel("Correlação com a nota final")
    ax.set_title("Correlação das variáveis com a nota final")
    fig.tight_layout()
    print(figura_para_ascii(fig))


def gerar_graficos(df, y_test, resultado):
    grafico_real_vs_previsto(np.array(y_test), resultado["y_pred"])
    grafico_distribuicao_erros(resultado["erros"], resultado["media_erro"])
    grafico_correlacao(df)


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

    print(f"\nResumo salvo em {OUT_DIR}/resumo_resultados.txt")


if __name__ == "__main__":
    main()
