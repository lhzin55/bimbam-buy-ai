# BimBam Buy AI

Projeto de assistente para PDFs usando Flask e Gemini.

## Estrutura

- `app.py` — aplicação principal
- `documentos/` — PDFs de entrada
- `.env` — chave Gemini
- `requirements.txt` — dependências mínimas

# Como instalar
Clone o repositório e acesse a pasta:

Bash
git clone <URL_DO_REPOSITORIO>
cd bimbam-buy-ai
Crie e ative um ambiente virtual:

Bash
python -m venv venv
# No Linux/macOS:
source venv/bin/activate
# No Windows:
venv\Scripts\activate
Instale as dependências:

Bash
pip install -r requirements.txt
Configure as variáveis de ambiente:
Crie um arquivo .env na raiz do projeto com as seguintes chaves:

Snippet de código
GEMINI_API_KEY=sua_chave_aqui
GEMINI_MODEL=gemini-flash-latest

## Como usar

1. Coloque seus PDFs dentro de `documentos/`.
2. Crie um arquivo `.env` com:

```env
GOOGLE_API_KEY=...
# ou
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-flash-latest
```

3. Ative o virtualenv e rode:

```bash
python app.py
```

4. Abra `http://127.0.0.1:5000`.

## Dependências mínimas

- Flask
- python-dotenv
- google-genai
- langchain-community
- langchain-text-splitters
- pypdf