import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from google import genai

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ===============================================================================
# CONFIGURAÇÃO
# ===============================================================================

load_dotenv()

app = Flask(__name__)

ROOT_DIR = Path(__file__).resolve().parent
DOCUMENTOS_DIR = ROOT_DIR / "documentos"

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")

PROMPT_BIMBAM_BUY = """
Você é o assistente virtual oficial da BimBam Buy.

Sua função é responder perguntas utilizando exclusivamente as informações presentes no contexto fornecido.

Regras obrigatórias:

- Nunca invente informações.
- Nunca utilize conhecimento próprio.
- Se a resposta não estiver no contexto, diga claramente que ela não foi encontrada nos documentos.
- Sempre responda em português.
- Seja objetivo.
- Quando possível, cite o documento utilizado.
- Não mencione que recebeu chunks ou contexto.
- Não faça suposições.
- Não misture informações externas.
"""

STOPWORDS = {
    "a", "o", "os", "as", "de", "do", "da", "dos", "das", "e", "em", "um", "uma",
    "para", "por", "com", "sem", "no", "na", "nos", "nas", "que", "qual", "quais",
    "como", "quando", "onde", "posso", "pode", "sobre", "me", "meu", "minha",
    "ser", "sao", "são", "é", "ao", "aos", "à", "às", "ou", "se", "isso", "essa",
    "esse", "essa", "isto", "este", "esta", "isso", "isso", "tem", "ter", "há"
}

# ===============================================================================
# ESTRUTURA DE DADOS
# ===============================================================================

@dataclass
class Trecho:
    texto: str
    fonte: str
    pagina: int | None
    tokens: set[str]

BASE_CONHECIMENTO: list[Trecho] | None = None

# ===============================================================================
# FUNÇÕES DE TEXTO
# ===============================================================================

def normalizar_texto(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto)
    texto = texto.encode("ASCII", "ignore").decode("ASCII")
    return texto.lower()


def tokenizar(texto: str) -> list[str]:
    texto = normalizar_texto(texto)
    return re.findall(r"[a-z0-9]+", texto)

# ===============================================================================
# CARREGAMENTO DOS DOCUMENTOS
# ===============================================================================

def carregar_pdfs() -> list:
    if not DOCUMENTOS_DIR.exists():
        raise FileNotFoundError(
            "A pasta 'documentos' não foi encontrada. Crie a pasta e coloque os PDFs dentro dela."
        )

    arquivos = sorted(DOCUMENTOS_DIR.glob("*.pdf"))
    if not arquivos:
        raise FileNotFoundError(
            "Nenhum PDF encontrado na pasta 'documentos'. Coloque os PDFs da BimBam Buy lá dentro."
        )

    documentos = []
    for arquivo in arquivos:
        print(f"Carregando: {arquivo.name}")
        loader = PyPDFLoader(str(arquivo))
        documentos.extend(loader.load())

    return documentos


def dividir_documentos(documentos: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return splitter.split_documents(documentos)


def construir_base_conhecimento() -> list[Trecho]:
    documentos = carregar_pdfs()
    chunks = dividir_documentos(documentos)

    base = []
    for chunk in chunks:
        fonte = Path(chunk.metadata.get("source", "documento")).name
        pagina = chunk.metadata.get("page")
        pagina = pagina + 1 if pagina is not None else None

        base.append(
            Trecho(
                texto=chunk.page_content,
                fonte=fonte,
                pagina=pagina,
                tokens=set(tokenizar(chunk.page_content)),
            )
        )

    return base


def obter_base_conhecimento() -> list[Trecho]:
    global BASE_CONHECIMENTO

    if BASE_CONHECIMENTO is None:
        BASE_CONHECIMENTO = construir_base_conhecimento()

    return BASE_CONHECIMENTO

# ===============================================================================
# BUSCA SIMPLES NOS DOCUMENTOS
# ===============================================================================

def pontuar_trecho(pergunta_tokens: list[str], trecho: Trecho) -> int:
    tokens_pergunta = [t for t in pergunta_tokens if t not in STOPWORDS]
    if not tokens_pergunta:
        return 0

    compartilhados = set(tokens_pergunta) & trecho.tokens
    return len(compartilhados)


def buscar_trechos_relevantes(pergunta: str, k: int = 4) -> list[Trecho]:
    base = obter_base_conhecimento()
    pergunta_tokens = tokenizar(pergunta)

    if not base:
        return []

    pontuados = []
    for trecho in base:
        score = pontuar_trecho(pergunta_tokens, trecho)
        if score > 0:
            pontuados.append((score, trecho))

    pontuados.sort(key=lambda x: x[0], reverse=True)

    if pontuados:
        return [item[1] for item in pontuados[:k]]

    return base[:k]


def montar_contexto(trechos: list[Trecho]) -> str:
    blocos = []

    for trecho in trechos:
        fonte = trecho.fonte
        if trecho.pagina is not None:
            blocos.append(
                f"Fonte: {fonte} | Página: {trecho.pagina}\n{trecho.texto}"
            )
        else:
            blocos.append(
                f"Fonte: {fonte}\n{trecho.texto}"
            )

    return "\n\n---\n\n".join(blocos)

# ===============================================================================
# GEMINI
# ===============================================================================

def obter_api_key() -> str:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Variável GOOGLE_API_KEY ou GEMINI_API_KEY não configurada no .env."
        )
    return api_key


def gerar_resposta_gemini(pergunta: str, contexto: str) -> str:
    api_key = obter_api_key()
    client = genai.Client(api_key=api_key)

    prompt = f"""
{PROMPT_BIMBAM_BUY}

Contexto dos documentos:
{contexto}

Pergunta do usuário:
{pergunta}
"""

    resposta = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )

    texto = (resposta.text or "").strip()
    if not texto:
        raise RuntimeError("O Gemini retornou uma resposta vazia.")

    return texto


def responder_pergunta(pergunta: str) -> dict:
    if not pergunta.strip():
        return {
            "resposta": "Digite uma pergunta antes de enviar.",
            "fontes": [],
        }

    trechos = buscar_trechos_relevantes(pergunta, k=4)
    contexto = montar_contexto(trechos)

    if not contexto.strip():
        return {
            "resposta": "Não encontrei informações suficientes nos documentos para responder essa pergunta.",
            "fontes": [],
        }

    try:
        resposta = gerar_resposta_gemini(pergunta, contexto)
    except Exception as e:
        return {
            "resposta": f"Erro ao consultar o Gemini: {e}",
            "fontes": [],
        }

    fontes = []
    for trecho in trechos:
        if trecho.pagina is not None:
            fonte = f"{trecho.fonte} — página {trecho.pagina}"
        else:
            fonte = trecho.fonte
        if fonte not in fontes:
            fontes.append(fonte)

    return {
        "resposta": resposta,
        "fontes": fontes,
    }

@app.route("/", methods=["GET", "POST"])
def home():
    pergunta = ""
    resposta = None
    fontes = []
    erro = None

    try:
        obter_base_conhecimento()
    except Exception as e:
        erro = str(e)

    if request.method == "POST":
        pergunta = request.form.get("pergunta", "").strip()

        if pergunta:
            resultado = responder_pergunta(pergunta)
            resposta = resultado["resposta"]
            fontes = resultado["fontes"]
        else:
            erro = "Digite uma pergunta antes de enviar."

    return render_template(
        "index.html",
        pergunta=pergunta,
        resposta=resposta,
        fontes=fontes,
        erro=erro,
    )

@app.route("/api/perguntar", methods=["POST"])
def api_perguntar():
    dados = request.get_json(silent=True) or {}
    pergunta = (dados.get("pergunta") or "").strip()

    if not pergunta:
        return jsonify({"erro": "Campo 'pergunta' é obrigatório."}), 400

    resultado = responder_pergunta(pergunta)
    return jsonify(resultado), 200

# ===============================================================================
# MAIN
# ===============================================================================

if __name__ == "__main__":
    porta = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=porta, debug=False)
