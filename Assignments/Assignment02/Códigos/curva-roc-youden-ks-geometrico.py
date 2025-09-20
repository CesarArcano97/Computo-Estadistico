import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score

# --- 1. Datos ---
edad = np.array([20, 23, 24, 25, 25, 26, 26, 28, 28, 29, 30, 30, 30, 30, 30, 32, 32, 33, 33,
                 34, 34, 34, 34, 34, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 39, 39, 39, 40, 40, 41,
                 41, 42, 42, 42, 42, 43, 43, 43, 44, 44, 44, 45, 45, 45, 46, 46, 47, 47, 47, 47, 48,
                 48, 48, 49, 49, 49, 50, 50, 51, 52, 52, 53, 53, 54, 55, 55, 55, 56, 56, 56, 57, 57,
                 57, 57, 57, 57, 58, 58, 58, 59, 59, 60, 60, 61, 62, 62, 63, 64, 64, 64, 65, 69])
coro = np.array([0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,1,0,1,
                 0,0,0,0,1,0,1,0,0,1,1,0,0,1,1,1,0,1,1,1,1,0,0,1,1,1,0,1,1,0,1,1,0,0,1,1,1,1,0,1,1,
                 1,1,0,1,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1])

# --- 2. Modelo de Regresión Logística ---
X = edad.reshape(-1, 1)
y = coro

logit = LogisticRegression(solver="lbfgs")
logit.fit(X, y)
phat = logit.predict_proba(X)[:, 1]  # Probabilidades para la clase 1

# --- 3. Función Auxiliar para Métricas ROC ---
def compute_roc_metrics(y_true, scores, positive_label=1):
    """Calcula métricas ROC y puntos de corte óptimos."""
    if positive_label == 0:
        # Invertir para analizar la clase 0 como positiva
        y_true = 1 - y_true
        scores = 1 - scores

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    # Puntos de corte óptimos
    youden_idx = np.argmax(tpr - fpr)
    ks_idx = np.argmax(np.abs(tpr - fpr))
    dist_idx = np.argmin(np.sqrt((fpr - 0)**2 + (tpr - 1)**2))

    # Métricas en cada punto óptimo
    results = {}
    optimal_points = {
        "Youden": thresholds[youden_idx],
        "KS": thresholds[ks_idx],
        "Dist": thresholds[dist_idx]
    }
    
    for name, thr in optimal_points.items():
        y_pred = (scores >= thr).astype(int)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        results[name] = (thr, acc, f1)

    return fpr, tpr, roc_auc, youden_idx, ks_idx, dist_idx, results

# --- 4. Cálculo de Métricas ---
fpr1, tpr1, auc1, y_idx1, ks_idx1, d_idx1, res1 = compute_roc_metrics(y, phat, positive_label=1)
fpr0, tpr0, auc0, y_idx0, ks_idx0, d_idx0, res0 = compute_roc_metrics(y, phat, positive_label=0)

# --- 5. Reporte en Consola ---
print("--- MÉTRICAS PARA CLASE 1 (coro=1) ---")
for k, v in res1.items():
    print(f"Óptimo por {k}: thr={v[0]:.3f}, Accuracy={v[1]:.3f}, F1={v[2]:.3f}")

print("\n--- MÉTRICAS PARA CLASE 0 (coro=0) ---")
for k, v in res0.items():
    print(f"Óptimo por {k}: thr={v[0]:.3f}, Accuracy={v[1]:.3f}, F1={v[2]:.3f}")

# --- 6. Visualizaciones ---

# Gráfico 1: Youden y KS
plt.figure(figsize=(8, 8))
plt.plot(fpr1, tpr1, color="darkorange", lw=2, label=f'ROC Clase 1 (AUC = {auc1:.2f})')
plt.plot(fpr0, tpr0, color="blue", lw=2, label=f'ROC Clase 0 (AUC = {auc0:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)

plt.scatter(fpr1[y_idx1], tpr1[y_idx1], c="red", s=120, marker="X", label="Youden (Clase 1)")
plt.scatter(fpr1[ks_idx1], tpr1[ks_idx1], c="purple", s=80, marker="o", label="KS (Clase 1)")
plt.scatter(fpr0[y_idx0], tpr0[y_idx0], c="green", s=120, marker="X", label="Youden (Clase 0)")
plt.scatter(fpr0[ks_idx0], tpr0[ks_idx0], c="cyan", s=80, marker="o", label="KS (Clase 0)")

plt.title("Curva ROC: Puntos Óptimos de Youden y KS")
plt.xlabel("Tasa de Falsos Positivos (1 - Especificidad)")
plt.ylabel("Tasa de Verdaderos Positivos (Sensibilidad)")
plt.legend(loc="lower right")
plt.grid(True)
plt.axis('equal')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.show()


# Gráfico 2: Distancia a (0,1)
plt.figure(figsize=(8, 8))
plt.plot(fpr1, tpr1, color="darkorange", lw=2, label=f'ROC Clase 1 (AUC = {auc1:.2f})')
plt.plot(fpr0, tpr0, color="blue", lw=2, label=f'ROC Clase 0 (AUC = {auc0:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)

plt.scatter(0, 1, c="black", s=150, marker="*", label="Punto Ideal (0,1)")
plt.scatter(fpr1[d_idx1], tpr1[d_idx1], c="red", s=100, label="Dist. Mínima (Clase 1)")
plt.plot([0, fpr1[d_idx1]], [1, tpr1[d_idx1]], 'r--')
plt.scatter(fpr0[d_idx0], tpr0[d_idx0], c="green", s=100, label="Dist. Mínima (Clase 0)")
plt.plot([0, fpr0[d_idx0]], [1, tpr0[d_idx0]], 'g--')

plt.title("Curva ROC: Distancia Mínima al Punto Ideal (0,1)")
plt.xlabel("Tasa de Falsos Positivos (1 - Especificidad)")
plt.ylabel("Tasa de Verdaderos Positivos (Sensibilidad)")
plt.legend(loc="lower right")
plt.grid(True)
plt.axis('equal')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.show()