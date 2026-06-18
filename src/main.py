from datasets.loader import Loader
from treinamento.treino_rbf import TreinadorRBF

X = Loader.carregar_csv(
    "data/processed/treino.csv"
)

X_entrada = [linha[:-1] for linha in X]
y = [linha[-1] for linha in X]

rbf = TreinadorRBF(
    n_centros=10
)

rbf.treinar(
    X_entrada,
    y
)

print("Treinamento concluído")