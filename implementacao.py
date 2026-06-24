"""
Rede Neural RBF - PP05
Optativa III - Redes Neurais Artificiais
UEMG Divinópolis

Implementação do zero conforme pseudocódigo fornecido.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ============================================================
# DADOS
# ============================================================

# Dados de treinamento
X_train = np.array([
    [0.2563, 0.9503], [0.2405, 0.9018], [0.1157, 0.3676], [0.5147, 0.0167],
    [0.4127, 0.3275], [0.2809, 0.5830], [0.8263, 0.9301], [0.9359, 0.8724],
    [0.1096, 0.9165], [0.5158, 0.8545], [0.1334, 0.1362], [0.6371, 0.1439],
    [0.7052, 0.6277], [0.8703, 0.8666], [0.2612, 0.6109], [0.0244, 0.5279],
    [0.9588, 0.3672], [0.9332, 0.5499], [0.9623, 0.2961], [0.7297, 0.5776],
    [0.4560, 0.1871], [0.1715, 0.7713], [0.5571, 0.5485], [0.3344, 0.0259],
    [0.4803, 0.7635], [0.9721, 0.4850], [0.8318, 0.7844], [0.1373, 0.0292],
    [0.3660, 0.8581], [0.3626, 0.7302], [0.6474, 0.3324], [0.3461, 0.2398],
    [0.1353, 0.8120], [0.3463, 0.1017], [0.9086, 0.1947], [0.5227, 0.2321],
    [0.5153, 0.2041], [0.1832, 0.0661], [0.5015, 0.9812], [0.5024, 0.5274],
])

d_train = np.array([
    -1, -1,  1,  1,  1,  1, -1, -1, -1, -1,
     1,  1, -1, -1,  1,  1, -1, -1, -1, -1,
     1,  1, -1,  1, -1, -1, -1,  1, -1, -1,
     1,  1,  1,  1, -1,  1,  1,  1, -1, -1,
])

# Dados de validação (d extraídos da Tabela 3 do PDF)
X_val = np.array([
    [0.8705, 0.9329], [0.0388, 0.2703], [0.8236, 0.4458], [0.7075, 0.1502],
    [0.9587, 0.8663], [0.6115, 0.9365], [0.3534, 0.3646], [0.3268, 0.2766],
    [0.6129, 0.4518], [0.9948, 0.4962],
])

d_val = np.array([-1, 1, -1, 1, -1, -1, 1, 1, -1, -1])

N_train = X_train.shape[0]  # 40 amostras
n_inputs = 2                 # x1, x2
n1 = 2                       # neurônios na camada oculta (2 gaussianas)
n_outputs = 1                # 1 saída

# ============================================================
# ESTÁGIO 1 - TREINAMENTO DA CAMADA INTERMEDIÁRIA (K-MEANS)
# ============================================================
print("=" * 60)
print("ESTÁGIO 1 - K-MEANS (Camada Intermediária)")
print("=" * 60)

# <2> Iniciar os pesos com as 2 PRIMEIRAS amostras de treinamento
W1 = X_train[:n1, :].copy()   # shape (2, 2): W1[j] = centro do neurônio j
print(f"\nCentros iniciais (2 primeiras amostras):")
print(f"  W1[0] = {W1[0]}")
print(f"  W1[1] = {W1[1]}")

# <3> Repetir até convergência
iteration = 0
while True:
    # <3.1> Para todas as amostras, atribuir ao cluster mais próximo
    clusters = {j: [] for j in range(n1)}   # Omega^(j)

    for k in range(N_train):
        x_k = X_train[k]
        # <3.1.1> Calcular distâncias euclidianas
        distances = []
        for j in range(n1):
            diff = x_k - W1[j]
            dist = np.sqrt(np.sum(diff ** 2))
            distances.append(dist)
        # <3.1.2> Selecionar neurônio j com menor distância
        j_min = int(np.argmin(distances))
        # <3.1.3> Atribuir x^(k) ao grupo Omega^(j_min)
        clusters[j_min].append(k)

    # <3.2> Atualizar centros: média das amostras de cada cluster
    W1_new = W1.copy()
    for j in range(n1):
        if len(clusters[j]) > 0:
            indices = clusters[j]
            # W_ji^(1) = (1/m^(j)) * sum_{x^(k) in Omega^(j)} x^(k)
            W1_new[j] = np.mean(X_train[indices], axis=0)

    iteration += 1

    # Verificar convergência: sem mudanças nos grupos
    changed = not np.allclose(W1, W1_new)
    W1 = W1_new.copy()

    if not changed:
        print(f"\nK-means convergiu em {iteration} iterações.")
        break

print(f"\nClusters finais:")
for j in range(n1):
    print(f"  Cluster {j+1}: {len(clusters[j])} amostras -> índices {clusters[j]}")

print(f"\nCentros finais (W1):")
for j in range(n1):
    print(f"  Cluster {j+1}: W1[{j}] = [{W1[j][0]:.6f}, {W1[j][1]:.6f}]")

# <4> Calcular variância de cada gaussiana (distância quadrática média)
sigma2 = np.zeros(n1)
for j in range(n1):
    indices = clusters[j]
    m_j = len(indices)
    if m_j > 0:
        s = 0.0
        for k in indices:
            x_k = X_train[k]
            # sum_{i=1}^{n} (x_i^(k) - W_ji^(1))^2
            s += np.sum((x_k - W1[j]) ** 2)
        sigma2[j] = s / m_j

print(f"\nVariâncias (sigma^2):")
for j in range(n1):
    print(f"  sigma2[{j+1}] = {sigma2[j]:.6f}")

print(f"\n{'='*60}")
print("Tabela 1 - Clusters da Camada Escondida")
print("="*60)
print(f"{'Cluster':<10} {'Centro':<35} {'Variância':<15}")
print("-"*60)
for j in range(n1):
    centro_str = f"[{W1[j][0]:.4f}, {W1[j][1]:.4f}]"
    print(f"  {j+1:<8} {centro_str:<35} {sigma2[j]:.6f}")


# ============================================================
# FUNÇÃO GAUSSIANA (equação 6.2)
# g_j^(1) = exp(-||x - W_j^(1)||^2 / (2 * sigma_j^2))
# ============================================================
def gaussian(x, center, var):
    """Função de ativação gaussiana."""
    diff = x - center
    dist2 = np.sum(diff ** 2)
    return np.exp(-dist2 / (2.0 * var))


def compute_z(x):
    """Calcula pseudo-amostra z = [g1, g2] para entrada x."""
    z = np.zeros(n1)
    for j in range(n1):
        z[j] = gaussian(x, W1[j], sigma2[j])
    return z


# ============================================================
# ESTÁGIO 2 - TREINAMENTO DA CAMADA DE SAÍDA (Regra Delta)
# ============================================================
print(f"\n{'='*60}")
print("ESTÁGIO 2 - Regra Delta (Camada de Saída)")
print("="*60)

eta = 0.01       # taxa de aprendizagem
epsilon = 1e-7   # precisão requerida

# <3> Iniciar W2 com valores aleatórios pequenos
np.random.seed(42)
W2 = np.random.uniform(-0.01, 0.01, size=(n1,))   # pesos da camada de saída
theta = float(np.random.uniform(-0.01, 0.01))      # bias (threshold)

print(f"\nPesos iniciais (aleatórios):")
print(f"  W2 = {W2}")
print(f"  theta = {theta:.6f}")

# <5> Pré-calcular pseudo-amostras z^(k) para todas as amostras
Z_train = np.zeros((N_train, n1))
for k in range(N_train):
    Z_train[k] = compute_z(X_train[k])

# <6> Iniciar contador de épocas
epoca = 0
historico_erro = []

# Calcular erro inicial E_M (equação 5.8): E_M = (1/2) * sum_k (d^(k) - y^(k))^2
def calcular_Em(W2, theta):
    total = 0.0
    for k in range(N_train):
        z_k = Z_train[k]
        # <3.2> u_j^(2) = sum_j W_ji^(2) * z_j + theta  (equação 6.3)
        u = np.dot(W2, z_k) + theta
        # <3.3> y = u (ativação linear, equação 6.4)
        y = u
        total += (d_train[k] - y) ** 2
    return 0.5 * total

E_M_anterior = calcular_Em(W2, theta)
historico_erro.append(E_M_anterior)

# <7> Repetir
max_epocas = 100000
print(f"\nTreinando...")
while True:
    # <7.2> Para todos os pares (z^(k), d^(k)), atualizar pesos (Regra Delta)
    for k in range(N_train):
        z_k = Z_train[k]
        d_k = d_train[k]

        # Forward pass
        u = np.dot(W2, z_k) + theta
        y = u  # ativação linear: f'(u) = 1

        # Erro local
        e_k = d_k - y

        # Atualização dos pesos (regra Delta generalizada, f'=1)
        # Delta W_ji^(2) = eta * e_k * z_j
        W2 += eta * e_k * z_k
        # Delta theta = eta * e_k * 1  (bias como peso com entrada fixa +1)
        theta += eta * e_k

    # <7.3> Calcular erro atual
    E_M_atual = calcular_Em(W2, theta)
    historico_erro.append(E_M_atual)

    # <7.4> Incrementar época
    epoca += 1

    # Critério de parada: |E_M_atual - E_M_anterior| <= epsilon
    if abs(E_M_atual - E_M_anterior) <= epsilon:
        print(f"  Convergiu na época {epoca}.")
        break

    if epoca >= max_epocas:
        print(f"  Limite de épocas atingido ({max_epocas}).")
        break

    E_M_anterior = E_M_atual

print(f"\nPesos finais da camada de saída:")
print(f"  W2[1,1] = {W2[0]:.6f}")
print(f"  W2[2,1] = {W2[1]:.6f}")
print(f"  theta_1  = {theta:.6f}")

print(f"\n{'='*60}")
print("Tabela 2 - Pesos da Camada de Saída")
print("="*60)
print(f"  W_(1,1)^(2) = {W2[0]:.6f}")
print(f"  W_(2,1)^(2) = {W2[1]:.6f}")
print(f"  theta_1      = {theta:.6f}")


# ============================================================
# FASE DE OPERAÇÃO - VALIDAÇÃO
# ============================================================
print(f"\n{'='*60}")
print("FASE DE OPERAÇÃO - Validação")
print("="*60)

def forward(x):
    """Propagação completa: entrada x -> saída y."""
    z = compute_z(x)                    # pseudo-amostras (gaussianas)
    u = np.dot(W2, z) + theta          # u^(2) (eq. 6.3)
    y = u                               # saída linear (eq. 6.4)
    return y, z

print(f"\n{'Sample':<8} {'x1':<8} {'x2':<8} {'d':<5} {'y':>10} {'y_pos':>7}")
print("-"*50)

y_val = []
y_pos_val = []
for i in range(len(X_val)):
    x = X_val[i]
    d = d_val[i]
    y, _ = forward(x)
    y_pos = 1 if y >= 0 else -1
    y_val.append(y)
    y_pos_val.append(y_pos)
    print(f"  {i+1:<6} {x[0]:<8.4f} {x[1]:<8.4f} {int(d):<5} {y:>10.6f} {int(y_pos):>7}")

y_val = np.array(y_val)
y_pos_val = np.array(y_pos_val)

# Taxa de acerto
acertos = np.sum(y_pos_val == d_val)
taxa_acerto = acertos / len(d_val) * 100
print(f"\n  Taxa de acerto: {acertos}/{len(d_val)} = {taxa_acerto:.1f}%")


# ============================================================
# MATRIZ DE CONFUSÃO E MÉTRICAS
# ============================================================
print(f"\n{'='*60}")
print("MATRIZ DE CONFUSÃO E MÉTRICAS")
print("="*60)

# Positivo = 1 (presença de radiação), Negativo = -1
TP = int(np.sum((y_pos_val == 1) & (d_val == 1)))
TN = int(np.sum((y_pos_val == -1) & (d_val == -1)))
FP = int(np.sum((y_pos_val == 1) & (d_val == -1)))
FN = int(np.sum((y_pos_val == -1) & (d_val == 1)))

N_acertos = TP + TN
N_erros   = FP + FN
total     = len(d_val)

Acuracia      = N_acertos / total
Sensibilidade = TP / (TP + FN) if (TP + FN) > 0 else 0.0
Especificidade = TN / (TN + FP) if (TN + FP) > 0 else 0.0
Precisao      = TP / (TP + FP) if (TP + FP) > 0 else 0.0

print(f"\nMatriz de Confusão:")
print(f"                    Predito +1   Predito -1")
print(f"  Real +1 (Exist.)    TP={TP:<4}      FN={FN}")
print(f"  Real -1 (Inex.)     FP={FP:<4}      TN={TN}")

print(f"\nMétricas:")
print(f"  N_acertos     = {N_acertos}")
print(f"  N_erros       = {N_erros}")
print(f"  Acurácia      = {Acuracia:.4f}  ({Acuracia*100:.1f}%)")
print(f"  Sensibilidade = {Sensibilidade:.4f}  ({Sensibilidade*100:.1f}%)")
print(f"  Especificidade= {Especificidade:.4f}  ({Especificidade*100:.1f}%)")
print(f"  Precisão      = {Precisao:.4f}  ({Precisao*100:.1f}%)")


# ============================================================
# GRÁFICOS
# ============================================================
fig = plt.figure(figsize=(16, 12))
fig.suptitle("Rede Neural RBF - PP05\nReconhecimento de Presença de Radiação",
             fontsize=14, fontweight='bold')

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# ---- Gráfico 1: Erro quadrático médio por época ----
ax1 = fig.add_subplot(gs[0, :])
epocas_eixo = np.arange(len(historico_erro))
ax1.plot(epocas_eixo, historico_erro, 'b-', linewidth=1.5, label='$E_M$ por época')
ax1.set_xlabel("Época", fontsize=12)
ax1.set_ylabel("Erro Quadrático Médio $E_M$", fontsize=12)
ax1.set_title("Estágio 2 - Evolução do Erro Quadrático Médio", fontsize=12)
ax1.grid(True, alpha=0.4)
ax1.legend(fontsize=11)
ax1.set_xlim(0, len(historico_erro)-1)
final_err = historico_erro[-1]
ax1.annotate(f'Erro final: {final_err:.6f}\nÉpocas: {epoca}',
             xy=(epoca, final_err),
             xytext=(epoca*0.6, max(historico_erro)*0.7),
             fontsize=10,
             arrowprops=dict(arrowstyle='->', color='red'),
             color='red')

# ---- Gráfico 2: Dados de treinamento e centros dos clusters ----
ax2 = fig.add_subplot(gs[1, 0])
colors_train = ['green' if d == 1 else 'red' for d in d_train]
for i in range(N_train):
    marker = '^' if d_train[i] == 1 else 'o'
    ax2.scatter(X_train[i, 0], X_train[i, 1], c=colors_train[i],
                marker=marker, s=60, alpha=0.7)
# Centros dos clusters
for j in range(n1):
    ax2.scatter(W1[j, 0], W1[j, 1], c='blue', marker='*', s=300,
                zorder=5, label=f'Centro {j+1}: [{W1[j,0]:.3f}, {W1[j,1]:.3f}]')
ax2.set_xlabel("$x_1$", fontsize=12)
ax2.set_ylabel("$x_2$", fontsize=12)
ax2.set_title("Dados de Treinamento e Centros K-Means", fontsize=11)
ax2.legend(fontsize=9, loc='upper right')
# Legenda manual
from matplotlib.lines import Line2D
leg_elements = [
    Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=10, label='d=+1 (Radiação)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red',   markersize=10, label='d=-1 (Sem radiação)'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='blue',  markersize=14, label='Centro K-Means'),
]
ax2.legend(handles=leg_elements, fontsize=8, loc='upper right')
ax2.grid(True, alpha=0.3)

# ---- Gráfico 3: Matriz de confusão ----
ax3 = fig.add_subplot(gs[1, 1])
conf_matrix = np.array([[TP, FN], [FP, TN]])
im = ax3.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
ax3.set_xticks([0, 1])
ax3.set_yticks([0, 1])
ax3.set_xticklabels(['Predito +1\n(Radiação)', 'Predito -1\n(Sem Rad.)'], fontsize=10)
ax3.set_yticklabels(['Real +1\n(Radiação)', 'Real -1\n(Sem Rad.)'], fontsize=10)
ax3.set_title("Matriz de Confusão (Validação)", fontsize=11)
labels = [['TP', 'FN'], ['FP', 'TN']]
values = [[TP, FN], [FP, TN]]
for i in range(2):
    for j in range(2):
        ax3.text(j, i, f'{labels[i][j]}={values[i][j]}',
                 ha='center', va='center', fontsize=14, fontweight='bold',
                 color='black')
plt.colorbar(im, ax=ax3)

# Caixa de métricas
metrics_text = (
    f"Acurácia:       {Acuracia*100:.1f}%\n"
    f"Sensibilidade:  {Sensibilidade*100:.1f}%\n"
    f"Especificidade: {Especificidade*100:.1f}%\n"
    f"Precisão:       {Precisao*100:.1f}%\n"
    f"N_acertos: {N_acertos} | N_erros: {N_erros}"
)
fig.text(0.5, 0.01, metrics_text, ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.savefig("PP05_RBF_resultados.png",
            dpi=150, bbox_inches='tight')
plt.close()
print(f"\nGráfico salvo.")


# ============================================================
# RESUMO FINAL
# ============================================================
print(f"\n{'='*60}")
print("RESUMO FINAL")
print("="*60)
print(f"\nTabela 1 - Clusters:")
for j in range(n1):
    print(f"  Cluster {j+1}: Centro=[{W1[j,0]:.4f}, {W1[j,1]:.4f}]  Variância={sigma2[j]:.6f}")

print(f"\nTabela 2 - Pesos camada de saída:")
print(f"  W_(1,1)^(2) = {W2[0]:.6f}")
print(f"  W_(2,1)^(2) = {W2[1]:.6f}")
print(f"  theta_1      = {theta:.6f}")

print(f"\nTabela 3 - Validação:")
print(f"{'Sample':<8} {'x1':<8} {'x2':<8} {'d':<5} {'y':>10} {'y_pos':>7} {'Acerto?':>8}")
print("-"*58)
for i in range(len(X_val)):
    acerto = "✓" if y_pos_val[i] == d_val[i] else "✗"
    print(f"  {i+1:<6} {X_val[i,0]:<8.4f} {X_val[i,1]:<8.4f} {int(d_val[i]):<5} "
          f"{y_val[i]:>10.6f} {int(y_pos_val[i]):>7} {acerto:>8}")
print(f"\n  Taxa de acerto: {taxa_acerto:.1f}%")
print(f"  Total épocas:   {epoca}")