# treinador_excel.py
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split

# Configurações / caminhos
INPUT_XLSX = Path("treinamento_gerado.xlsx")
SHEET_TREINO = "treinamento"
ARTIFACT_DIR = Path("model_artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)
MODEL_OUT = ARTIFACT_DIR / "modelo_cabem.pkl"
METADATA_OUT = ARTIFACT_DIR / "metadata.json"
SAVE_CONF_MATRIX = True  # alterar para False para não salvar figura

# 1) Carregar dados do Excel
print("📦 Carregando", INPUT_XLSX, "sheet:", SHEET_TREINO)
if not INPUT_XLSX.exists():
    raise FileNotFoundError(f"{INPUT_XLSX} não encontrado. Gere o Excel primeiro.")

df = pd.read_excel(INPUT_XLSX, sheet_name=SHEET_TREINO)
print("✅ Sheet carregada. Linhas:", len(df))

# 2) Normalizar nomes das colunas esperadas
df.columns = [c.strip() for c in df.columns]
expected_cols = ['produto', 'c_brin', 'l_brin', 'a_brin', 'area_brin',
                 'c_gar', 'l_gar', 'a_gar', 'area_gar', 'cabe_area',
                 'cabe_dim', 'label']
# aceitar se tiver pelo menos as colunas essenciais
essenciais = ['produto', 'c_brin', 'l_brin', 'a_brin', 'c_gar', 'l_gar', 'a_gar', 'label']
for c in essenciais:
    if c not in df.columns:
        raise ValueError(f"Coluna essencial ausente no sheet '{SHEET_TREINO}': {c}")

# 3) Limpeza mínima e tipos
df['produto'] = df['produto'].astype(str).str.strip()
for col in ['c_brin', 'l_brin', 'a_brin', 'area_brin', 'c_gar', 'l_gar', 'a_gar', 'area_gar']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
# remover linhas com NaN nas colunas essenciais
df = df.dropna(subset=['produto', 'c_brin', 'l_brin', 'a_brin', 'c_gar', 'l_gar', 'a_gar', 'label'])
df['label'] = df['label'].astype(int)

print("📊 Dataset após limpeza:", df.shape)

# 4) One-hot encode do produto (mantemos todas as categorias encontradas)
df_encoded = pd.get_dummies(df, columns=['produto'], prefix='produto', drop_first=False)

# 5) Definir features e target
# incluir também area_brin e area_gar se presentes
feature_cols = ['c_brin', 'l_brin', 'a_brin', 'c_gar', 'l_gar', 'a_gar']
if 'area_brin' in df_encoded.columns:
    feature_cols.append('area_brin')
if 'area_gar' in df_encoded.columns:
    feature_cols.append('area_gar')

# adicionar as colunas one-hot de produto
produto_cols = [c for c in df_encoded.columns if c.startswith('produto_')]
feature_cols += produto_cols

X = df_encoded[feature_cols]
y = df_encoded['label']

print("🔎 Features usadas (exemplo):", feature_cols[:8], " ... total:", len(feature_cols))
print("Classes no target:", y.value_counts().to_dict())

# 6) Split treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None
)

# 7) Treinar modelo
print("🛠️ Treinando RandomForest...")
model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 8) Avaliar
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n🎯 Acurácia: {acc:.4f}\n")
print("📋 Relatório de classificação:")
print(classification_report(y_test, y_pred, zero_division=0))

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
print("Matriz de confusão:\n", cm)

if SAVE_CONF_MATRIX:
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Matriz de Confusão")
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.tight_layout()
    fig_path = ARTIFACT_DIR / "confusion_matrix.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"✅ Matriz de confusão salva em {fig_path}")

# 9) Salvar modelo e metadata
joblib.dump(model, MODEL_OUT)
metadata = {
    "feature_columns": feature_cols,
    "produto_columns": produto_cols,
    "n_samples": int(len(df)),
    "random_state": 42
}
with open(METADATA_OUT, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print(f"\n✅ Modelo salvo em {MODEL_OUT}")
print(f"✅ Metadata salva em {METADATA_OUT}")