# script pra criar uma base de alunos fake, ja que nao temos dados reais pra usar
# a ideia e: cada aluno tem horas de estudo, frequencia, % de atividades entregues
# e a nota da avaliacao anterior. com isso a gente calcula a nota final dele

import numpy as np
import pandas as pd

SEED = 42  # fixei uma seed pra sempre gerar os mesmos dados (facilita testar)
N_ALUNOS = 400


def gerar_base(n_alunos=N_ALUNOS, seed=SEED):
    rng = np.random.default_rng(seed)

    # gerando os valores de cada aluno de forma aleatoria, dentro de faixas que fazem sentido
    horas_estudo = rng.uniform(0, 20, n_alunos)
    frequencia = rng.uniform(40, 100, n_alunos)
    atividades_entregues = rng.uniform(0, 100, n_alunos)
    nota_anterior = rng.uniform(0, 10, n_alunos)

    # aqui monta a nota final como uma soma ponderada das variaveis
    # os pesos foram no chute mesmo, tentando deixar algo parecido com a realidade
    # (nota anterior e horas de estudo pesam mais que o resto)
    nota_base = (
        0.18 * horas_estudo
        + 0.03 * frequencia
        + 0.02 * atividades_entregues
        + 0.35 * nota_anterior
    )

    # sem ruido a nota ficaria "perfeita demais", entao coloquei uma variacao aleatoria
    # pra simular que tem outras coisas que influenciam a nota que a gente nao mediu
    ruido = rng.normal(0, 0.8, n_alunos)
    nota_final = nota_base + ruido
    nota_final = np.clip(nota_final, 0, 10)  # garantindo que fica entre 0 e 10

    df = pd.DataFrame({
        "aluno_id": np.arange(1, n_alunos + 1),
        "horas_estudo": horas_estudo.round(2),
        "frequencia": frequencia.round(2),
        "atividades_entregues": atividades_entregues.round(2),
        "nota_anterior": nota_anterior.round(2),
        "nota_final": nota_final.round(2),
    })

    return df


if __name__ == "__main__":
    df = gerar_base()
    df.to_csv("data/alunos.csv", index=False)
    print(f"base gerada com {len(df)} alunos, salva em data/alunos.csv")
    print(df.head())
