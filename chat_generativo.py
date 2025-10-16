# chat_strict_category_final.py
import re
import json
import joblib
import unicodedata
import difflib
import pandas as pd
from pathlib import Path

# RapidFuzz opcional
try:
    from rapidfuzz import fuzz, process as rf_process
    _HAS_RAPIDFUZZ = True
except Exception:
    _HAS_RAPIDFUZZ = False

# Configuração
BRINQUEDOS_CSV = "medidas_brinquedos.csv"
MODEL_PATH = Path("model_artifacts") / "modelo_cabem.pkl"
METADATA_PATH = Path("model_artifacts") / "metadata.json"
ENCODING = "ISO-8859-1"
FUZZY_MIN_SCORE = 60
CONFIG_AUTO_ACCEPT_SCORE = 30
TOP_CANDIDATES_SHOW = 3

STOPWORDS = set([
    "uma","um","o","a","os","as","em","no","na","de","do","dos","das","pra","para",
    "com","sem","por","e","ou","meu","minha","seu","sua","este","esta","aquele","aquela"
])

# -------------------------
# Normalização e utilitários
# -------------------------
def normalize_text_full(s):
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in s)
    s = " ".join(s.split())
    return s

def tokens_nonstop(s):
    return [t for t in normalize_text_full(s).split() if t and t not in STOPWORDS]

def token_overlap_score(q, name):
    tq = set(tokens_nonstop(q))
    tn = set(tokens_nonstop(name))
    if not tq or not tn:
        return 0.0
    inter = tq.intersection(tn)
    score = 100.0 * (len(inter) / max(len(tq), len(tn)))
    return score

# -------------------------
# Carregar catálogo e artefatos
# -------------------------
def carregar_brinquedos(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    df = pd.read_csv(path, sep=";", encoding=ENCODING, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    if "produto" not in df.columns and "nome" in df.columns:
        df = df.rename(columns={"nome": "produto"})
    for col in list(df.columns):
        if any(x in col.lower() for x in ["comprimento", "largura", "altura", "area"]):
            df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["produto_norm"] = df["produto"].astype(str).map(normalize_text_full)
    df["produto_tokens"] = df["produto_norm"].map(lambda s: set(s.split()))
    return df

def carregar_modelo_e_metadata(model_path, metadata_path):
    if not Path(model_path).exists() or not Path(metadata_path).exists():
        raise FileNotFoundError("Modelo ou metadata não encontrados. Rode o treinador primeiro.")
    model = joblib.load(model_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    feature_columns = metadata.get("feature_columns", [])
    return model, feature_columns

# -------------------------
# Matching helpers
# -------------------------
def extract_numeric_token(query):
    m = re.search(r'(\d+(?:[.,]\d+)?)\s*m\b', query, flags=re.IGNORECASE)
    if m:
        return m.group(1).replace(',', '.')
    return None

def extract_category_tokens(query):
    toks = tokens_nonstop(query)
    return [t for t in toks if not re.fullmatch(r'\d+(?:[.,]\d+)?', t)]

def filter_candidates_by_num_and_cat(df, num_token, cat_tokens):
    dfc = df.copy()
    mask_num = pd.Series([False]*len(dfc), index=dfc.index)
    mask_cat = pd.Series([False]*len(dfc), index=dfc.index)
    if num_token:
        num_plain = num_token.replace('.', ' ')
        mask_num = dfc['produto_norm'].str.contains(re.escape(num_token), na=False) | dfc['produto_norm'].str.contains(re.escape(num_plain), na=False)
    if cat_tokens:
        mask_cat = dfc['produto_tokens'].apply(lambda s: any(ct in s for ct in cat_tokens))
    # prioridade: num+cat, cat only, num only, all
    df_num_cat = dfc[mask_num & mask_cat]
    if not df_num_cat.empty:
        return df_num_cat
    df_cat = dfc[mask_cat]
    if not df_cat.empty:
        return df_cat
    df_num = dfc[mask_num]
    if not df_num.empty:
        return df_num
    return dfc

def find_best_brinquedo_freeform(query, catalog_df, min_score=FUZZY_MIN_SCORE, top_n=5):
    q_norm = normalize_text_full(query)
    if "produto_norm" not in catalog_df.columns:
        catalog_df = catalog_df.copy()
        catalog_df["produto_norm"] = catalog_df["produto"].astype(str).map(normalize_text_full)
        catalog_df["produto_tokens"] = catalog_df["produto_norm"].map(lambda s: set(s.split()))
    names_norm = catalog_df["produto_norm"].tolist()
    names_orig = catalog_df["produto"].tolist()

    # token overlap
    best_idx = None
    best_score = -1
    best_method = None
    for i, n in enumerate(names_norm):
        score = token_overlap_score(q_norm, n)
        if score > best_score:
            best_score = score
            best_idx = i
            best_method = "token_overlap"
    if best_score >= min_score:
        return names_orig[best_idx], int(best_score), best_method

    # substring direct
    for i, n in enumerate(names_norm):
        if n and (n in q_norm or q_norm in n):
            return names_orig[i], 100, "substring"

    # rapidfuzz
    if _HAS_RAPIDFUZZ:
        choices = {n: i for i, n in enumerate(names_norm)}
        results = rf_process.extract(q_norm, choices.keys(), scorer=fuzz.token_set_ratio, limit=top_n)
        if results:
            best_name, score, _ = results[0]
            idx = choices[best_name]
            return names_orig[idx], int(score), "rapidfuzz_token_set"

    # difflib fallback
    matches = difflib.get_close_matches(q_norm, names_norm, n=top_n, cutoff=0.0)
    if matches:
        best = matches[0]
        score = int(difflib.SequenceMatcher(None, q_norm, best).ratio() * 100)
        idx = names_norm.index(best)
        return names_orig[idx], score, "difflib"

    if best_idx is not None:
        return names_orig[best_idx], int(best_score), best_method
    return None, 0, None

def top_candidates(query, catalog_df, k=3):
    q_norm = normalize_text_full(query)
    if "produto_norm" not in catalog_df.columns:
        catalog_df = catalog_df.copy()
        catalog_df["produto_norm"] = catalog_df["produto"].astype(str).map(normalize_text_full)
        catalog_df["produto_tokens"] = catalog_df["produto_norm"].map(lambda s: set(s.split()))
    names_norm = catalog_df["produto_norm"].tolist()
    names_orig = catalog_df["produto"].tolist()
    results = []
    if _HAS_RAPIDFUZZ:
        choices = {n: i for i, n in enumerate(names_norm)}
        res = rf_process.extract(q_norm, choices.keys(), scorer=fuzz.token_set_ratio, limit=k)
        for name, score, _ in res:
            idx = choices[name]
            results.append((names_orig[idx], int(score)))
        return results
    for i, n in enumerate(names_norm):
        score_tok = token_overlap_score(q_norm, n)
        score_seq = int(difflib.SequenceMatcher(None, q_norm, n).ratio() * 100)
        combined = max(score_tok, score_seq)
        results.append((names_orig[i], int(combined)))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:k]

# -------------------------
# Extração de dimensões (heurística)
# -------------------------
dim_pattern_multiples = re.compile(r'(\d+(?:[.,]\d+)?)\s*[x×]\s*(\d+(?:[.,]\d+)?)(?:\s*[x×]\s*(\d+(?:[.,]\d+)?))?', re.IGNORECASE)
single_number_m_pattern = re.compile(r'(\d+(?:[.,]\d+)?)\s*(m2|m²)\b', re.IGNORECASE)
single_number_any_m = re.compile(r'(\d+(?:[.,]\d+)?)\s*m\b', re.IGNORECASE)

def parse_dimensions_from_text(text):
    """
    Extrai dimensões da garagem da frase preferindo o número mais próximo de uma palavra-chave
    de espaço (ex.: 'garagem', 'garagem de', 'sala', 'salão'). Suporta também padrões LxW[xH]
    e área (m2). Retorna (c_gar, l_gar, a_gar) ou None.
    """
    text = str(text)

    # 1) padrão LxW[xH] primeiro
    m = dim_pattern_multiples.search(text)
    if m:
        parts = [p.replace(",", ".") for p in m.groups() if p]
        vals = [float(p) for p in parts]
        if len(vals) == 2:
            return vals[0], vals[1], 2.5
        elif len(vals) == 3:
            return vals[0], vals[1], vals[2]

    # 2) área explícita m2
    m2 = single_number_m_pattern.search(text)
    if m2:
        area = float(m2.group(1).replace(",", "."))
        side = round(area ** 0.5, 3)
        return side, side, 2.5

    # 3) encontrar todos tokens com número + "m"
    matches = list(re.finditer(r'(\d+(?:[.,]\d+)?)\s*m\b', text, flags=re.IGNORECASE))
    if not matches:
        return None

    # preparar índices das ocorrências das palavras-chave de espaço
    space_keywords = ['garagem', 'garagem de', 'garagens', 'salão', 'sala', 'salon', 'garage', 'box', 'estacionamento']
    lowered = text.lower()

    # coletar posições (start index) das keywords
    kw_positions = []
    for kw in space_keywords:
        for mkw in re.finditer(re.escape(kw), lowered):
            kw_positions.append(mkw.start())
    # se não encontrou keyword, manter comportamento anterior: usar último número
    if not kw_positions:
        last = matches[-1]
        val = float(last.group(1).replace(",", "."))
        side = round(val ** 0.5, 3)
        return side, side, 2.5

    # Para cada numeric match, calcular a distância mínima até qualquer keyword (em caracteres)
    best_match = None
    best_dist = None
    for match in matches:
        mstart = match.start()
        # distância mínima para todas as keywords
        dist = min(abs(mstart - kp) for kp in kw_positions)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_match = match
        elif dist == best_dist:
            # se empate, preferir o match que aparece depois (último)
            if match.start() > best_match.start():
                best_match = match

    # usar o melhor match encontrado (mais próximo da keyword)
    val = float(best_match.group(1).replace(",", "."))
    # heurística: quando número muito grande pode ser comprimento; aqui mantemos interpretação como área (m²) conforme fluxo atual
    side = round(val ** 0.5, 3)
    return side, side, 2.5


# -------------------------
# Montar features conforme metadata
# -------------------------
def montar_features(produto_nome, c_brin, l_brin, a_brin, c_gar, l_gar, a_gar, feature_columns):
    area_brin = float(c_brin) * float(l_brin)
    area_gar = float(c_gar) * float(l_gar)
    base = {
        "c_brin": float(c_brin),
        "l_brin": float(l_brin),
        "a_brin": float(a_brin),
        "area_brin": float(area_brin),
        "c_gar": float(c_gar),
        "l_gar": float(l_gar),
        "a_gar": float(a_gar),
        "area_gar": float(area_gar),
    }
    row = {c: 0 for c in feature_columns}
    for k, v in base.items():
        if k in row:
            row[k] = v
    prod_col = f"produto_{normalize_text_full(produto_nome)}"
    if prod_col in row:
        row[prod_col] = 1
    else:
        for c in feature_columns:
            if c.startswith("produto_") and normalize_text_full(produto_nome) in c.lower():
                row[c] = 1
                break
    return pd.DataFrame([row])[feature_columns]

# -------------------------
# Responder pergunta (rigorosa: exige categoria quando presente)
# -------------------------
def responder_pergunta(pergunta_text, df_brinquedos, model, feature_columns):
    texto = pergunta_text.strip()
    dims = parse_dimensions_from_text(texto)
    cleaned = re.sub(r'garagem|garagens|cab(e|er|em)|cabe|entra|caber|dentro|um|uma|em|na|no|de|do|pra|para', ' ', texto, flags=re.IGNORECASE)
    cleaned = re.sub(r'[^\w\s,\.x×]', ' ', cleaned)
    cleaned = " ".join(cleaned.split())

    num_token = extract_numeric_token(texto)
    cat_tokens = extract_category_tokens(texto)

    # Se o usuário indicou uma categoria, garantir que exista ao menos um produto da categoria no catálogo.
    if cat_tokens:
        df_cat_all = df_brinquedos[df_brinquedos['produto_tokens'].apply(lambda s: any(ct in s for ct in cat_tokens))]
        if df_cat_all.empty:
            return "Não encontrei produtos correspondentes à categoria mencionada. Por favor, verifique o nome."
    else:
        df_cat_all = pd.DataFrame([])

    # construir subconjunto preferencial (num+cat > cat only > num only > all)
    df_pref = filter_candidates_by_num_and_cat(df_brinquedos, num_token, cat_tokens)

    # se houve tokens de categoria, restringir imediatamente ao conjunto de produtos da categoria
    if cat_tokens:
        df_search = df_cat_all
    else:
        df_search = df_pref

    # buscar melhor candidato estritamente no conjunto permitido
    best, score, method = find_best_brinquedo_freeform(cleaned, df_search)

    if best is None:
        # mostrar candidatos top do conjunto permitido para transparência
        cand = top_candidates(cleaned, df_search if not df_search.empty else df_brinquedos, k=TOP_CANDIDATES_SHOW)
        cand_str = ", ".join([f"{c[0]} ({c[1]}%)" for c in cand])
        return f"Não identifiquei um produto da categoria solicitada com confiança suficiente. Candidatos: {cand_str}. Por favor, reformule."

    # exigir score mínimo de aceitação automática
    if score < CONFIG_AUTO_ACCEPT_SCORE:
        cand = top_candidates(cleaned, df_search if not df_search.empty else df_brinquedos, k=TOP_CANDIDATES_SHOW)
        cand_str = ", ".join([f"{c[0]} ({c[1]}%)" for c in cand])
        return f"Não identifiquei com confiança suficiente (score={score}%). Candidatos: {cand_str}. Por favor, reformule."

    # obter dimensões do catálogo
    row = df_brinquedos[df_brinquedos["produto"].astype(str) == best].iloc[0]
    def pegar(colnames):
        for c in colnames:
            if c in row and pd.notna(row[c]):
                return float(row[c])
        return None
    c_brin = pegar(["comprimento(m)", "c_brin", "comprimento"])
    l_brin = pegar(["largura(m)", "l_brin", "largura"])
    a_brin = pegar(["altura(m)", "a_brin", "altura"])
    if None in (c_brin, l_brin, a_brin):
        return "Dimensões do brinquedo incompletas no catálogo."

    if dims:
        c_gar, l_gar, a_gar = dims
    else:
        m2 = re.search(r'(\d+(?:[.,]\d+)?)\s*(m2|m²)', texto, re.IGNORECASE)
        if m2:
            area = float(m2.group(1).replace(",", "."))
            side = area ** 0.5
            c_gar, l_gar, a_gar = side, side, 2.5
        else:
            c_gar, l_gar, a_gar = 5.0, 4.0, 2.5

    X = montar_features(best, c_brin, l_brin, a_brin, c_gar, l_gar, a_gar, feature_columns)
    pred = model.predict(X)[0]

    area_b = round(c_brin * l_brin, 3)
    area_g = round(float(c_gar) * float(l_gar), 3)
    encaixa_orient1 = (c_brin <= float(c_gar) and l_brin <= float(l_gar))
    encaixa_orient2 = (c_brin <= float(l_gar) and l_brin <= float(c_gar))
    encaixa = encaixa_orient1 or encaixa_orient2
    altura_ok = a_brin <= float(a_gar)
    cabe_rule = bool(area_g >= area_b and encaixa and altura_ok)

    if cabe_rule:
        resposta = (
            f"Resposta: Sim — a '{best}' cabe no espaço informado. "
            f"Resumo: área necessária {area_b} m²; área disponível {area_g} m²; "
            f"encaixe possível: {encaixa}; altura adequada: {altura_ok}."
        )
    else:
        motivos = []
        if area_g < area_b:
            motivos.append(f"área insuficiente ({area_g} < {area_b} m²)")
        if not encaixa:
            motivos.append("dimensões não permitem encaixe (comprimento/largura)")
        if not altura_ok:
            motivos.append("altura insuficiente")
        motivos_text = "; ".join(motivos) if motivos else "não cabe por critérios internos"
        resposta = (
            f"Resposta: Não — a '{best}' não cabe neste espaço. Motivos: {motivos_text}. "
            f"Dados: área brinquedo {area_b} m²; área garagem {area_g} m²."
        )

    # nota compacta de confiança apenas se abaixo do ideal
    if score < FUZZY_MIN_SCORE:
        cand = top_candidates(cleaned, df_search if not df_search.empty else df_brinquedos, k=TOP_CANDIDATES_SHOW)
        cand_str = ", ".join([f"{c[0]}({c[1]}%)" for c in cand])
        resposta += f"\n\nObservação: correspondência moderada ({score}%). Outros candidatos na categoria: {cand_str}."

    resposta += f"\n\n(Identificado: {best}; confiança={score}%; método={method})"
    return resposta

# -------------------------
# Inicialização
# -------------------------
def main():
    df_b = carregar_brinquedos(BRINQUEDOS_CSV)
    model, feature_columns = carregar_modelo_e_metadata(MODEL_PATH, METADATA_PATH)
    print("Chat iniciado (regra rígida: exige categoria quando presente). Digite 'sair' para encerrar.")
    while True:
        q = input("\nPergunta: ").strip()
        if q.lower() in ("sair", "exit", "quit"):
            print("Encerrando.")
            break
        resposta = responder_pergunta(q, df_b, model, feature_columns)
        print("\n" + resposta)

if __name__ == "__main__":
    main()