import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Carregar a base de treino
df = pd.read_csv("base_treino_model.csv")

# 2. Tratar variáveis categóricas (One-Hot Encoding para sigla_partido e sigla_uf)
X = pd.get_dummies(df[['sigla_partido', 'sigla_uf_x', 'ano']], drop_first=True)
y = df['ideb']

# 3. Divisão Treino e Teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. BASELINE: Prever a média simples do treino para todas as amostras
y_pred_baseline = np.full_like(y_test, fill_value=y_train.mean())

# 5. MODELO: Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
y_pred_model = model.predict(X_test)

# 6. Avaliação de Métricas (RMSE, MAE, R²)
def calcular_metricas(y_real, y_pred, nome):
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    mae = mean_absolute_error(y_real, y_pred)
    r2 = r2_score(y_real, y_pred)
    print(f"--- {nome} ---")
    print(f"RMSE (Erro Quadrático Médio): {rmse:.4f}")
    print(f"MAE (Erro Absoluto Médio): {mae:.4f}")
    print(f"R² (Coeficiente de Determinação): {r2:.4f}\n")

print("\n=== RESULTADOS DE AVALIAÇÃO DO MODELO ===\n")
calcular_metricas(y_test, y_pred_baseline, "BASELINE (Média Simples)")
calcular_metricas(y_test, y_pred_model, "RANDOM FOREST REGRESSOR")

# 7. Importância das Variáveis (Feature Importance)
importances = model.feature_importances_
feature_names = X.columns
df_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
df_imp = df_imp.sort_values(by='importance', ascending=False)

print("\n=== TOP 15 VARIÁVEIS MAIS IMPORTANTES DA RANDOM FOREST ===")
print(df_imp.head(15).to_string(index=False))