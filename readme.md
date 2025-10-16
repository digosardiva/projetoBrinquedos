# 🧠 Chat Generativo — Brinquedos vs Espaço

Versão final do chat local para verificar se brinquedos cabem em espaços.  
Matching estrito por categoria, priorização de tokens numéricos e parser robusto para dimensões.

---

## 📘 Introdução
Este repositório contém o chat `chat generativo.py` — versão final que faz **matching estrito por categoria**, **prioriza tokens numéricos** quando aplicável e **avalia se um brinquedo cabe em um espaço** usando o modelo salvo.  
O pipeline de treinamento e a geração do arquivo Excel já estão prontos e permanecem inalterados.

---

## ⚙️ Pré-requisitos
- Python 3.10 ou 3.11 (testado)
- pip instalado
- Ollama local.
- Arquivos obrigatórios:
  - `medidas_brinquedos.csv` — catálogo de produtos
  - `model_artifacts/modelo_cabem.pkl` — modelo treinado
  - `model_artifacts/metadata.json` — metadados (feature_columns)

---

## 🚀 Instalação
1. Instale o Ollama na sua maquina

2. Rode o comando no seu terminal para baixar o llama
   ```powershell
    ollama pull llama3.2
   ```
3. Criar e ativar ambiente virtual
   - **Windows**
     ```powershell
     python -m venv .venv
     .venv\Scripts\activate
     ```
   - **macOS / Linux**
     ```bash
     python -m venv .venv
     source .venv/bin/activate
     ```

2. Instalar dependências
   ```bash
   pip install -r requirements.txt
   ```

--

**Observações:**
- `rapidfuzz` é opcional, mas recomendado para melhor *fuzzy matching*  
- Se não quiser instalar, o código usa fallback com `difflib` (builtin)  
- Ajuste versões conforme sua política de compatibilidade

---

## 📂 Estrutura esperada

```
chat generativo.py          # Script principal
medidas_brinquedos.csv      # Catálogo CSV com coluna "produto" e dimensões
model_artifacts/
  ├── modelo_cabem.pkl      # Modelo treinado
  └── metadata.json         # Metadados de features
requirements.txt            # Dependências Python
```

---

## 💬 Como executar

1. Verifique se todos os arquivos obrigatórios estão no diretório do projeto.

2. Execute o gerador do modelo de treinamento:
   ```bash
   python .\geraExcel.py
   ```
3. Execute o treinamento do modelo:
    Ele irá retornar algumas estatisticas.
   ```bash
   python .\treinador.py
   ```

4. Execute o chat:
   ```bash
   python "chat generativo.py"
   ```
5. Faça perguntas como:
   ```
   uma piscina de bolinhas 1m cabe numa garagem de 10m?
   cama elástica 1,80m cabe em uma garagem de 15m?
   ```
6. Para sair:
   ```
   sair
   ```

---

## 🧩 Comportamento do chat

- Entrada em linguagem natural: o chat extrai automaticamente **brinquedo e dimensões**.  
- Matching estrito por categoria: busca palavras-chave no nome do produto.  
- Prioridade numérica: identifica o número mais próximo de palavras como “garagem” para calcular o espaço.  
- Avaliação determinística: cálculo local com dimensões do catálogo.  
- Fallbacks: quando não há correspondência exata, o chat sugere opções próximas.

---

## ⚙️ Parâmetros ajustáveis (no topo do script)

| Parâmetro | Função |
|------------|--------|
| `FUZZY_MIN_SCORE` | Score mínimo considerado "bom" |
| `CONFIG_AUTO_ACCEPT_SCORE` | Limiar mínimo para aceitação automática |
| `TOP_CANDIDATES_SHOW` | Quantos candidatos exibir em sugestões |

> 💡 Dica: instalar `rapidfuzz` melhora o *matching*:  
> ```bash
> pip install rapidfuzz
> ```

---

## 📊 Boas práticas para o CSV

- Coluna com nome do produto: **produto** (ou **nome**)  
- Colunas numéricas recomendadas:
  - `comprimento(m)`
  - `largura(m)`
  - `altura(m)`
- Use **ponto** como separador decimal (o script converte vírgulas automaticamente)

---

## 🧯 Troubleshooting

| Problema | Solução |
|-----------|----------|
| **Arquivo não encontrado** | Verifique caminhos e nomes (`medidas_brinquedos.csv` e `model_artifacts/*`) |
| **Matching impreciso** | Revise nomes no CSV; instale `rapidfuzz` ou ajuste `FUZZY_MIN_SCORE` |
| **Dimensão interpretada incorretamente** | Peça a heurística pronta para tratar valores como comprimento linear |
