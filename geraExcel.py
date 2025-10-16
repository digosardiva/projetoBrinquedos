# gera_treinamento_excel.py
import random
import pandas as pd
from pathlib import Path
import unicodedata
import json

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

INPUT_BRINQ = Path("medidas_brinquedos.csv")
OUT_XLSX = Path("treinamento_gerado.xlsx")

# Leitura e normalização do catálogo
def carregar_brinquedos(path, encoding="ISO-8859-1"):
    df = pd.read_csv(path, sep=';', encoding=encoding, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    # Renomear coluna produto se necessário
    if 'produto' not in df.columns and 'nome' in df.columns:
        df = df.rename(columns={'nome': 'produto'})
    # Normalizar e converter numéricos
    for col in df.columns:
        if any(x in col.lower() for x in ['comprimento', 'largura', 'altura', 'area']):
            df[col] = df[col].astype(str).str.replace(',', '.')
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # Criar colunas padronizadas
    def norm_name(s):
        s = str(s).strip()
        s = unicodedata.normalize('NFKD', s)
        return s
    df['produto'] = df['produto'].astype(str).map(norm_name)
    # Colunas padronizadas
    df = df.rename(columns={
        col: col for col in df.columns
    })
    # Mapear para nomes curtos
    mapping = {}
    for c in df.columns:
        lc = c.lower()
        if 'comprimento' in lc: mapping['c_brin'] = c
        if 'largura' in lc: mapping['l_brin'] = c
        if 'altura' in lc: mapping['a_brin'] = c
        if 'area' in lc: mapping['area_brin'] = c
        if 'valor' in lc and 'aluguel' in lc: mapping['valorAluguel'] = c
    # Criar DataFrame com colunas esperadas
    df_out = pd.DataFrame()
    df_out['produto'] = df['produto']
    df_out['c_brin'] = df.get(mapping.get('c_brin')).astype(float)
    df_out['l_brin'] = df.get(mapping.get('l_brin')).astype(float)
    df_out['a_brin'] = df.get(mapping.get('a_brin')).astype(float)
    if mapping.get('area_brin'):
        df_out['area_brin'] = df.get(mapping.get('area_brin')).astype(float)
    else:
        df_out['area_brin'] = df_out['c_brin'] * df_out['l_brin']
    if mapping.get('valorAluguel'):
        df_out['valorAluguel'] = pd.to_numeric(df.get(mapping.get('valorAluguel')), errors='coerce')
    else:
        df_out['valorAluguel'] = pd.NA
    # Drop linhas inválidas
    df_out = df_out.dropna(subset=['c_brin','l_brin','a_brin','area_brin'])
    df_out = df_out.reset_index(drop=True)
    return df_out

# Gerar garagens: grid + amostras aleatórias
def gerar_garagens(grid_lengths=None, grid_widths=None, grid_heights=None, n_random=200):
    if grid_lengths is None:
        grid_lengths = [3,4,5,6,7,8,9,10,12,15,18,20]
    if grid_widths is None:
        grid_widths = [2,3,4,5,6,7,8,10,12,15,18,20]
    if grid_heights is None:
        grid_heights = [2.0, 2.2, 2.5, 3.0, 4.0, 5.0]
    garagens = []
    for L in grid_lengths:
        for W in grid_widths:
            for H in grid_heights:
                garagens.append((float(L), float(W), float(H)))
    for _ in range(n_random):
        L = round(random.uniform(2.0, 20.0), 2)
        W = round(random.uniform(2.0, 20.0), 2)
        H = round(random.uniform(2.0, 5.0), 2)
        garagens.append((L,W,H))
    # criar DataFrame
    dfg = pd.DataFrame(garagens, columns=['c_gar','l_gar','a_gar'])
    dfg['area_gar'] = dfg['c_gar'] * dfg['l_gar']
    dfg = dfg.drop_duplicates().reset_index(drop=True)
    return dfg

# Construir dataset de treinamento
def construir_treinamento(df_brinquedos, df_garagens):
    rows = []
    for _, br in df_brinquedos.iterrows():
        for _, g in df_garagens.iterrows():
            c_b, l_b, a_b = float(br['c_brin']), float(br['l_brin']), float(br['a_brin'])
            area_b = float(br['area_brin'])
            c_g, l_g, a_g = float(g['c_gar']), float(g['l_gar']), float(g['a_gar'])
            area_g = float(g['area_gar'])
            cabe_area = int(area_g >= area_b)
            encaixa_orient1 = (c_b <= c_g and l_b <= l_g)
            encaixa_orient2 = (c_b <= l_g and l_b <= c_g)
            cabe_dim = int((encaixa_orient1 or encaixa_orient2) and (a_b <= a_g))
            label = int(cabe_area and cabe_dim)
            rows.append({
                'produto': br['produto'],
                'c_brin': c_b,
                'l_brin': l_b,
                'a_brin': a_b,
                'area_brin': area_b,
                'c_gar': c_g,
                'l_gar': l_g,
                'a_gar': a_g,
                'area_gar': area_g,
                'cabe_area': cabe_area,
                'cabe_dim': cabe_dim,
                'label': label
            })
    df_train = pd.DataFrame(rows)
    return df_train

def main():
    df_b = carregar_brinquedos(INPUT_BRINQ)
    df_g = gerar_garagens(n_random=300)
    df_train = construir_treinamento(df_b, df_g)
    # metadata
    metadata = {
        "random_seed": RANDOM_SEED,
        "n_brinquedos": len(df_b),
        "n_garagens": len(df_g),
        "n_pairs": len(df_train),
        "feature_columns": [
            "c_brin","l_brin","a_brin","area_brin","c_gar","l_gar","a_gar","area_gar"
        ]  # as colunas principais; one-hot de produto fica a cargo do trainer
    }
    # Salvar Excel com múltiplas sheets
    with pd.ExcelWriter(OUT_XLSX, engine='openpyxl') as writer:
        df_b.to_excel(writer, sheet_name="brinquedos", index=False)
        df_g.to_excel(writer, sheet_name="garagens", index=False)
        df_train.sample(frac=1, random_state=RANDOM_SEED).to_excel(writer, sheet_name="treinamento", index=False)
        pd.DataFrame([metadata]).to_excel(writer, sheet_name="metadata", index=False)
    # Salvar também metadata json
    Path("model_artifacts").mkdir(exist_ok=True)
    with open(Path("model_artifacts/metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Arquivo salvo: {OUT_XLSX}  | pares gerados: {len(df_train)}")

if __name__ == "__main__":
    main()