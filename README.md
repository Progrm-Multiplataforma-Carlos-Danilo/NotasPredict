# Previsão de Nota Final com Machine Learning

Projeto acadêmico que gera uma base fake de alunos e treina um modelo de
Machine Learning para prever a nota final com base em horas de estudo,
frequência, atividades entregues e nota anterior. Também calcula média,
desvio padrão e intervalo de erro das previsões, com os gráficos de análise
impressos direto no terminal (ASCII).

## Stack

- Python 3.12
- pandas / numpy
- scikit-learn (Regressão Linear e Random Forest)

## Como executar

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

cd src
python gerar_dados.py      # gera data/alunos.csv
python treinar_modelo.py   # treina o modelo, calcula métricas e mostra os gráficos no terminal
```

Resumo estatístico salvo em `outputs/resumo_resultados.txt`.

## Integrantes

- Carlos Eduardo Fernandes Farias
- Danilo Santos Soares
