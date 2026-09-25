"""
Treina um modelo de regressão para prever a nota final dos alunos e avalia
seu desempenho com métricas estatísticas: média e desvio padrão dos erros,
e o intervalo de erro (IC 95%) das previsões.

Os gráficos de análise são impressos diretamente no terminal (ASCII),
sem gerar arquivos de imagem.
"""

import os

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


def barra(valor, valor_max, largura=40):
    """Desenha uma barra ASCII proporcional a valor/valor_max."""
    n = int(round(largura * valor / valor_max)) if valor_max else 0
    n = max(0, min(largura, n))
    return "#" * n


def grafico_real_vs_previsto(y_test, y_pred):
    print("\n--- Nota real vs. Nota prevista (amostra de 20 alunos) ---")
    idx = np.linspace(0, len(y_test) - 1, min(20, len(y_test))).astype(int)
    y_test_arr = np.array(y_test)
    largura = 30
    for i in idx:
        real = y_test_arr[i]
        prev = y_pred[i]
        linha = list(" " * (largura + 1))
        pos_real = min(largura, max(0, round(real / 10 * largura)))
        pos_prev = min(largura, max(0, round(prev / 10 * largura)))
        linha[pos_real] = "R"
        linha[pos_prev] = "P" if pos_prev != pos_real else "X"
        print(f"{real:4.1f} |{''.join(linha)}| previsto {prev:4.1f}")
    print("      (R = nota real, P = nota prevista, X = coincidem)")


def grafico_distribuicao_erros(erros, media_erro):
    print("\n--- Distribuição dos erros de previsão (histograma) ---")
    n_bins = 10
    minimo, maximo = erros.min(), erros.max()
    bins = np.linspace(minimo, maximo, n_bins + 1)
    contagem, _ = np.histogram(erros, bins=bins)
    max_contagem = contagem.max() if contagem.max() > 0 else 1
    for i in range(n_bins):
        faixa = f"[{bins[i]:5.2f}, {bins[i + 1]:5.2f})"
        print(f"{faixa} | {barra(contagem[i], max_contagem)} {contagem[i]}")
    print(f"Média do erro marcada em: {media_erro:.3f}")


def grafico_correlacao(df):
    print("\n--- Correlação de cada variável com a nota final ---")
    corr = df[FEATURES + [TARGET]].corr()[TARGET].drop(TARGET)
    for variavel, valor in corr.items():
        sinal = "+" if valor >= 0 else "-"
        print(f"{variavel:22s} {sinal}{abs(valor):.2f} {barra(abs(valor), 1.0, 30)}")


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
