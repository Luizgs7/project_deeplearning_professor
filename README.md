# 🧠 Neural Prof — Assistente de Deep Learning

Tutor interativo especializado em Deep Learning e IA, desenvolvido como aplicação de apoio pedagógico para a disciplina de Projetos de IA, do programa de especialização em IA Generativa, da Universidade Federal do Paraná, cursada em 2026. Conversa com o aluno via LLM, adapta as respostas ao nível de conhecimento e oferece recursos ativos de aprendizagem.

---

## ✨ Funcionalidades

| Funcionalidade | Descrição |
|---|---|
| 💬 **Chat com streaming** | Respostas geradas progressivamente via Chat Completions API |
| 🎓 **Nível adaptativo** | Iniciante, Intermediário e Avançado — ajustam o rigor técnico das respostas |
| 💡 **Follow-up automático** | 2–3 perguntas relacionadas geradas após cada resposta, clicáveis como chips |
| 📝 **Quiz adaptativo** | 3 questões com 5 alternativas baseadas nos temas da conversa, com gabarito e explicações |
| 🗂️ **Tópicos sugeridos** | Atalhos na sidebar para os principais temas do curso |

---

## 🛠️ Stack

- **[Streamlit](https://streamlit.io) `>=1.58`** — interface web e gerenciamento de estado
- **[openai](https://github.com/openai/openai-python) `>=2.41`** — cliente para Chat Completions API com streaming
- **[chatlas](https://github.com/posit-dev/chatlas) `>=0.18`** — chamadas utilitárias ao modelo (follow-up e quiz)
- **[python-dotenv](https://github.com/theskumar/python-dotenv) `>=1.2`** — carregamento de variáveis de ambiente

---

## 🚀 Como rodar

### Pré-requisitos

- Python 3.11+
- Chave de API NVIDIA — obtenha em [build.nvidia.com](https://build.nvidia.com)

### Instalação

```bash
# 1. Clone o repositório
git clone https://github.com/Luizgs7/project_deeplearning_professor.git
cd project_deeplearning_professor

# 2. Crie e ative um ambiente virtual
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows

# 3. Instale as dependências
pip install -r requirements.txt
```

### Configuração

Crie um arquivo `.env` na raiz do projeto:

```env
NVIDIA_API_KEY=<sua_chave_aqui>
```

> ⚠️ O arquivo `.env` já está no `.gitignore`. Nunca o versione.

### Execução

```bash
streamlit run src/app.py
```

Acesse em `http://localhost:8501`.

---

## ☁️ Implantação (Oracle Cloud)

1. Provisione uma VM Linux com Python 3.11+
2. Clone o repositório e instale as dependências com `pip install -r requirements.txt`
3. Configure o arquivo `.env` com a `NVIDIA_API_KEY`
4. Execute `streamlit run src/app.py`
5. Abra a porta **8501** no security group da VM
6. Acesse via `http://<ip-publico>:8501`

---

## 🏗️ Arquitetura

- O histórico de conversa é mantido em `st.session_state` e enviado como contexto completo a cada turno.
- O cliente `openai.OpenAI` é usado diretamente para o chat principal (streaming compatível com a NVIDIA API). Instâncias separadas de `chatlas.ChatOpenAI` tratam chamadas utilitárias (follow-up e quiz), isolando o histórico principal.
- Cada sessão recebe um `session_id` único; perguntas e respostas são registradas em `chat_history.log` com rotação de 50 MB para evitar quebra inesperada da aplicação.

---

## 💭 Lições aprendidas

- Separar credenciais do código (`dotenv` + `.gitignore`) é inegociável desde o início.
- A nova **Responses API** da OpenAI (usada pelo chatlas em modo streaming) é incompatível com endpoints de terceiros; o cliente `openai` direto resolve.
- Streamlit é produtivo para protótipos, mas exige atenção ao modelo de re-renderização para features como streaming e chips interativos.

---

## 🔭 Melhorias futuras

- Log estruturado em JSONL para análise por sessão
- Persistência do histórico em banco de dados
- Autenticação e suporte a múltiplos usuários simultâneos
- Exportação do histórico de conversa pelo aluno
