import requests
import joblib
import pandas as pd
import json

# Carrega o modelo
model = joblib.load("modelo_cabem.pkl")

# Função para verificar se o brinquedo cabe no carro/garagem
def cabe_no_carro(c_brin, l_brin, a_brin, c_gar, l_gar, a_gar):
    # Cria DataFrame com nomes de colunas usados no treino
    X = pd.DataFrame([{
        "c_brin": c_brin,
        "l_brin": l_brin,
        "a_brin": a_brin,
        "c_gar": c_gar,
        "l_gar": l_gar,
        "a_gar": a_gar
    }])
    pred = model.predict(X)[0]
    return "Cabe" if pred == 1 else "Não cabe"

# Função para chamar o Ollama com streaming
def perguntar_ollama(pergunta, resposta_modelo):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3.2",
        "prompt": f"O usuário perguntou: '{pergunta}'. O modelo respondeu: '{resposta_modelo}'. Explique de forma simpática e técnica."
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        return f"Erro de conexão com Ollama: {e}"

    # Cada linha da resposta é um JSON separado (streaming)
    resposta_final = ""
    for linha in response.text.splitlines():
        linha = linha.strip()
        if not linha:
            continue
        try:
            parte = json.loads(linha)
            resposta_final += parte.get("response", "")
        except Exception:
            continue

    return resposta_final.strip() if resposta_final else "Sem resposta"

# Exemplo de uso
if __name__ == "__main__":
    pergunta = "Um brinquedo de 1.2x0.8x1.0 cabe em uma garagem de 2x2x2?"
    resposta_modelo = cabe_no_carro(1.2, 0.8, 1.0, 2.0, 2.0, 2.0)
    resposta_final = perguntar_ollama(pergunta, resposta_modelo)
    print("🧠", resposta_final)
