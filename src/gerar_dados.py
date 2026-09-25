"""
Gera uma base fake de alunos.

Cada aluno tem:
- horas_estudo: horas de estudo semanais (0 a 20)
- frequencia: percentual de presença nas aulas (40 a 100)
- atividades_entregues: percentual de atividades entregues (0 a 100)
- nota_anterior: nota do período/avaliação anterior (0 a 10)
- nota_final: variável alvo, calculada a partir das features acima + ruído aleatório

A nota_final segue uma combinação linear plausível das features, com ruído
gaussiano para simular a imprevisibilidade do mundo real.
"""

import numpy as np
import pandas as pd

SEED = 42
N_ALUNOS = 400


def gerar_base(n_alunos: int = N_ALUNOS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    horas_estudo = rng.uniform(0, 20, n_alunos)
    frequencia = rng.uniform(40, 100, n_alunos)
    atividades_entregues = rng.uniform(0, 100, n_alunos)
    nota_anterior = rng.uniform(0, 10, n_alunos)

    # Combinação linear com pesos plausíveis (quanto cada fator "pesa" na nota final)
    nota_base = (
        0.18 * horas_estudo          # até 20h de estudo -> até ~3.6 pontos
        + 0.03 * frequencia          # até 100% de frequência -> até 3.0 pontos
        + 0.02 * atividades_entregues  # até 100% entregue -> até 2.0 pontos
        + 0.35 * nota_anterior       # nota anterior pesa bastante -> até ~3.5 pontos
    )

    ruido = rng.normal(0, 0.8, n_alunos)  # ruído para simular fatores não observados
    nota_final = np.clip(nota_base + ruido, 0, 10)

    df = pd.DataFrame(
        {
            "aluno_id": np.arange(1, n_alunos + 1),
            "horas_estudo": horas_estudo.round(2),
            "frequencia": frequencia.round(2),
            "atividades_entregues": atividades_entregues.round(2),
            "nota_anterior": nota_anterior.round(2),
            "nota_final": nota_final.round(2),
        }
    )
    return df


if __name__ == "__main__":
    df = gerar_base()
    df.to_csv("data/alunos.csv", index=False)
    print(f"Base gerada com {len(df)} alunos em data/alunos.csv")
    print(df.head())
