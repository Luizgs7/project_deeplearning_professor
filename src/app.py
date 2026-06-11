import os
import re
import uuid
import json
import logging
from logging.handlers import RotatingFileHandler

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from openai import OpenAI
from chatlas import ChatOpenAI

# =========================================================
# CONFIGURAÇÃO
# =========================================================

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = RotatingFileHandler(
    "chat_history.log",
    maxBytes=50 * 1024 * 1024,
    backupCount=0,
    encoding="utf-8",
)
handler.setFormatter(
    logging.Formatter("%(asctime)s %(levelname)s %(message)s")
)
logger.addHandler(handler)

# Cliente openai usado para streaming (chat completions padrão, compatível com NVIDIA)
openai_client = OpenAI(
    api_key=os.getenv("NVIDIA_API_KEY"),
    base_url="https://integrate.api.nvidia.com/v1",
)

_MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """
### 🤖 System Prompt: Assistente Professor de Deep Learning

**[Papel e Identidade]**
Você é um AI Assistente de Ensino Universitário estritamente especializado em Deep Learning e Inteligência Artificial. Seu objetivo principal é atuar como um tutor paciente, didático e altamente técnico, ajudando alunos a compreender conceitos complexos, desde os fundamentos de Redes Neurais até arquiteturas avançadas de Machine Learning. Você é uma inteligência artificial e não deve fingir ter sentimentos humanos ou experiências do mundo físico.

**[Área de Conhecimento Mapeada]**
Sua base de conhecimento ativa deve focar exclusivamente nos seguintes tópicos:
*   **Fundamentos de IA e ML:** Regressão, classificação, funções de custo, otimizadores (SGD, Adam), regularização, backpropagation.
*   **Redes Neurais:** Perceptrons, MLPs, CNNs, RNNs, LSTMs, Autoencoders.
*   **Modelos de Atenção e Transformers:** Arquitetura Transformer, self-attention, codificadores/decodificadores, BERT, GPT, T5.
*   **Modelos Fundacionais (Foundation Models):** LLMs, modelos multimodais, difusão, fine-tuning, RAG, RLHF.
*   **Prática e Ferramentas:** PyTorch, TensorFlow, Keras, Hugging Face, ambientes de treinamento, métricas de avaliação.

**[Diretrizes Pedagógicas e Tom de Resposta]**
*   **Didática:** Explique conceitos complexos usando analogias claras, mas sem perder o rigor matemático ou técnico.
*   **Nivelamento:** Adapte sua resposta ao nível do aluno. Se ele fizer uma pergunta básica, forneça a base. Se fizer uma pergunta avançada sobre arquitetura, aprofunde-se tecnicamente.
*   **Incentivo:** Valide as dúvidas do aluno e encoraje o pensamento crítico. Não dê apenas a resposta final, explique o "porquê".

**[Restrições Rigorosas de Escopo (Guardrails)]**
Você está estritamente limitado ao campo de Deep Learning, Machine Learning, Ciência de Dados e IA. É **terminantemente proibido** responder a qualquer pergunta ou fornecer informações que fujam desse contexto.
*   Se o usuário perguntar sobre tópicos gerais (como culinária, política, história, medicina, esportes, entretenimento geral, ou linguagens de programação que não estejam aplicadas a IA), você deve se recusar a responder.
*   **Mensagem padrão para desvios:** *"Como assistente de ensino de Deep Learning, meu conhecimento é focado exclusivamente em Inteligência Artificial, Redes Neurais e tópicos relacionados. Como posso ajudar você a entender melhor a IA hoje?"*
*   Não forneça conselhos de vida, opiniões pessoais ou suporte emocional. Redirecione o usuário para o foco da aula educacionalmente.

**[Formatação de Saída]**
*   Use títulos em negrito e listas de pontos para organizar conceitos complexos.
*   Ao fornecer fórmulas, utilize notação LaTeX matemática padrão se necessário.
*   Ao fornecer código, utilize blocos de código formatados com a linguagem adequada (geralmente Python)."""

# =========================================================
# TÓPICOS — botões de sugestão de perguntas
# =========================================================

TOPICS = [
    {
        "label": "🔢  Fundamentos de IA & ML",
        "question": (
            "Pode me explicar como funciona o backpropagation em uma rede neural MLP, "
            "incluindo a regra da cadeia e como os gradientes são calculados e propagados "
            "em cada camada?"
        ),
    },
    {
        "label": "🕸️  Redes Neurais",
        "question": (
            "Qual é a diferença entre CNN, RNN e LSTM? "
            "Em quais tipos de problemas cada arquitetura se destaca "
            "e quais são suas principais limitações?"
        ),
    },
    {
        "label": "🔍  Atenção & Transformers",
        "question": (
            "Como funciona o mecanismo de self-attention nos Transformers? "
            "Por que ele é mais eficiente que as RNNs para capturar "
            "dependências de longo alcance em sequências?"
        ),
    },
    {
        "label": "🤖  Foundation Models",
        "question": (
            "O que é RAG (Retrieval-Augmented Generation) e como ele é implementado "
            "para melhorar as respostas de um LLM com base em conhecimento externo?"
        ),
    },
    {
        "label": "🛠️  Ferramentas & Prática",
        "question": (
            "Como construir um pipeline de treinamento completo com PyTorch, "
            "incluindo definição do modelo, DataLoader, função de loss, "
            "otimizador e loop de treino com validação?"
        ),
    },
]

LEVELS = {
    "Iniciante": (
        "O aluno está no nível Iniciante. Use linguagem simples e acessível, "
        "muitas analogias do cotidiano e evite fórmulas matemáticas complexas. "
        "Priorize a intuição antes do rigor formal."
    ),
    "Intermediário": (
        "O aluno está no nível Intermediário. Equilibre intuição e rigor técnico. "
        "Introduza notação matemática quando necessário, sempre explicando cada termo."
    ),
    "Avançado": (
        "O aluno está no nível Avançado. Use rigor matemático completo com notação formal. "
        "Assuma familiaridade com cálculo, álgebra linear e probabilidade. "
        "Aprofunde detalhes de implementação e teoria sem simplificações."
    ),
}

# =========================================================
# HELPERS — instâncias utilitárias (não poluem o histórico principal)
# =========================================================

def _make_util_chat():
    return ChatOpenAI(
        model=_MODEL,
        api_key=os.getenv("NVIDIA_API_KEY"),
        base_url="https://integrate.api.nvidia.com/v1",
    )


def _get_followup_questions(user_question, answer):
    _chat = _make_util_chat()
    prompt = (
        f"Pergunta do aluno sobre Deep Learning: {user_question[:300]}\n"
        f"Resposta do professor: {answer[:600]}\n\n"
        "Gere exatamente 3 perguntas de follow-up curtas (máximo 12 palavras cada) "
        "em português brasileiro que aprofundem ou complementem o tema discutido. "
        'Responda APENAS com um JSON array de 3 strings, sem mais nada: '
        '["Pergunta 1?", "Pergunta 2?", "Pergunta 3?"]'
    )
    try:
        resp = _chat.chat(prompt, stream=False)
        text = str(resp).strip()
        m = re.search(r'\[.*?\]', text, re.DOTALL)
        if m:
            qs = json.loads(m.group())
            if isinstance(qs, list) and qs:
                return [str(q) for q in qs[:3]]
    except Exception as exc:
        logger.error("followup generation error: %s", exc)
    return []


def _generate_quiz(history):
    _chat = _make_util_chat()
    context = "\n".join(
        f"{m['role']}: {m['content'][:300]}"
        for m in history[-8:]
    )
    prompt = (
        f"Contexto da conversa sobre Deep Learning:\n{context}\n\n"
        "Crie um quiz educacional com EXATAMENTE 3 perguntas de múltipla escolha "
        "sobre os temas discutidos acima.\n"
        "Cada pergunta deve ter EXATAMENTE 5 alternativas.\n"
        "Responda SOMENTE com JSON válido, sem texto extra:\n"
        '{\n'
        '  "topic": "tema principal do quiz",\n'
        '  "questions": [\n'
        '    {\n'
        '      "id": 1,\n'
        '      "question": "Texto da pergunta?",\n'
        '      "alternatives": ["A) opção", "B) opção", "C) opção", "D) opção", "E) opção"],\n'
        '      "correct": "A",\n'
        '      "explanation": "Breve explicação da resposta correta."\n'
        '    }\n'
        '  ]\n'
        '}'
    )
    try:
        resp = _chat.chat(prompt, stream=False)
        text = str(resp).strip()
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            data = json.loads(m.group())
            if "questions" in data and isinstance(data["questions"], list):
                return data
    except Exception as exc:
        logger.error("quiz generation error: %s", exc)
    return None

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Neural Prof — Deep Learning",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CSS THEME
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

.stApp {
    background: linear-gradient(145deg, #EEF2FF 0%, #F5F0FF 50%, #EFF9FF 100%);
    min-height: 100vh;
}

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid #E2E8F0 !important;
    box-shadow: 4px 0 24px rgba(67, 97, 238, 0.07) !important;
}

.main .block-container {
    max-width: 960px;
    padding-top: 1.5rem;
}

/* ---- Header ---- */
.neural-header {
    background: linear-gradient(135deg, #4361EE 0%, #7B2FBE 55%, #0EA5E9 100%);
    border-radius: 24px;
    padding: 32px 36px;
    margin-bottom: 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 24px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 12px 40px rgba(67, 97, 238, 0.28);
}

.neural-header::before {
    content: '';
    position: absolute;
    top: -60%; right: -5%;
    width: 420px; height: 420px;
    background: radial-gradient(circle, rgba(255,255,255,0.07) 0%, transparent 65%);
    pointer-events: none;
}

.header-left { flex: 1; z-index: 1; }

.header-badge {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    color: rgba(255,255,255,0.95);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 5px 14px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.28);
    margin-bottom: 12px;
}

.header-title {
    color: white;
    font-size: 2rem;
    font-weight: 800;
    margin: 0 0 8px 0;
    line-height: 1.2;
    text-shadow: 0 2px 10px rgba(0,0,0,0.12);
}

.header-subtitle {
    color: rgba(255,255,255,0.82);
    font-size: 0.92rem;
    margin: 0 0 20px 0;
    line-height: 1.65;
    max-width: 420px;
}

.header-tags { display: flex; flex-wrap: wrap; gap: 7px; }

.header-tag {
    background: rgba(255,255,255,0.14);
    color: rgba(255,255,255,0.95);
    font-size: 11.5px;
    font-weight: 500;
    padding: 5px 13px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.22);
}

.header-right { flex: 0 0 auto; z-index: 1; }

@media (max-width: 768px) {
    .neural-header { flex-direction: column; }
    .header-right { display: none; }
    .header-title { font-size: 1.6rem; }
}

/* ---- Chat Messages ---- */
[data-testid="stChatMessage"] {
    background: rgba(255, 255, 255, 0.88) !important;
    border-radius: 18px !important;
    border: 1px solid rgba(199, 210, 254, 0.5) !important;
    box-shadow: 0 2px 16px rgba(67, 97, 238, 0.06) !important;
    margin-bottom: 14px !important;
    backdrop-filter: blur(6px) !important;
    transition: box-shadow 0.2s ease !important;
}

[data-testid="stChatMessage"]:hover {
    box-shadow: 0 6px 24px rgba(67, 97, 238, 0.11) !important;
}

/* ---- Chat Input ---- */
[data-testid="stChatInput"] > div {
    border-radius: 18px !important;
    border: 2px solid #C7D2FE !important;
    background: rgba(255,255,255,0.95) !important;
    box-shadow: 0 4px 20px rgba(67, 97, 238, 0.1) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

[data-testid="stChatInput"]:focus-within > div {
    border-color: #4361EE !important;
    box-shadow: 0 4px 28px rgba(67, 97, 238, 0.2) !important;
}

/* ---- Difficulty segmented control ---- */
[data-testid="stSidebar"] [data-testid="stSegmentedControl"] {
    width: 100% !important;
}

[data-testid="stSidebar"] [data-testid="stSegmentedControl"] > div {
    background: #F1F5F9 !important;
    border-radius: 12px !important;
    padding: 3px !important;
    width: 100% !important;
    border: 1px solid #E2E8F0 !important;
}

[data-testid="stSidebar"] [data-testid="stSegmentedControl"] button {
    border-radius: 9px !important;
    font-size: 12.5px !important;
    font-weight: 500 !important;
    color: #64748B !important;
    flex: 1 !important;
    padding: 7px 6px !important;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    transition: color 0.2s !important;
}

[data-testid="stSidebar"] [data-testid="stSegmentedControl"] button[aria-pressed="true"] {
    background: white !important;
    color: #4361EE !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 6px rgba(67, 97, 238, 0.18) !important;
}

/* ---- Topic suggestion buttons (sidebar, secondary type) ---- */
[data-testid="stSidebar"] [data-testid="baseButton-secondary"] {
    background: #F8FAFF !important;
    color: #1E293B !important;
    border: 1px solid #EEF2FF !important;
    border-radius: 14px !important;
    box-shadow: none !important;
    text-align: left !important;
    padding: 12px 15px !important;
    font-weight: 500 !important;
    font-size: 13.5px !important;
    width: 100% !important;
    transform: none !important;
    letter-spacing: 0 !important;
    white-space: normal !important;
    height: auto !important;
    line-height: 1.4 !important;
    transition: border-color 0.2s, background 0.2s, color 0.2s !important;
}

[data-testid="stSidebar"] [data-testid="baseButton-secondary"]:hover {
    background: #EEF2FF !important;
    border-color: #A5B4FC !important;
    color: #4361EE !important;
    transform: none !important;
    box-shadow: 0 2px 10px rgba(67, 97, 238, 0.1) !important;
}

/* ---- Limpar button (sidebar, primary type) ---- */
[data-testid="stSidebar"] [data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #F87171, #EF4444) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    box-shadow: 0 4px 14px rgba(239, 68, 68, 0.25) !important;
    padding: 11px 22px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    width: 100% !important;
    letter-spacing: 0.2px !important;
    transform: none !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}

[data-testid="stSidebar"] [data-testid="baseButton-primary"]:hover {
    background: linear-gradient(135deg, #EF4444, #DC2626) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 18px rgba(239, 68, 68, 0.35) !important;
}

[data-testid="stSidebar"] [data-testid="baseButton-primary"]:active {
    transform: translateY(0) !important;
}

/* ---- Sidebar brand ---- */
.sidebar-brand {
    text-align: center;
    padding: 4px 0 18px 0;
    border-bottom: 1px solid #EEF2FF;
    margin-bottom: 18px;
}

.sidebar-brand .sb-icon {
    font-size: 42px;
    line-height: 1;
    display: block;
    margin-bottom: 8px;
}

.sidebar-brand h2 {
    background: linear-gradient(135deg, #4361EE, #7B2FBE);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 1.35rem;
    font-weight: 800;
    margin: 0 0 4px 0;
}

.sidebar-brand p { color: #94A3B8; font-size: 12px; margin: 0; }

.topics-label {
    color: #94A3B8;
    font-size: 10.5px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin: 0 0 10px 0;
    display: block;
}

/* ---- Divider ---- */
hr {
    border: none !important;
    border-top: 1px solid #E2E8F0 !important;
    margin: 18px 0 !important;
}

/* ---- Spinner ---- */
.stSpinner > div { border-top-color: #4361EE !important; }

/* ---- Scrollbar ---- */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #C7D2FE; border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: #818CF8; }

/* ---- Markdown ---- */
[data-testid="stMarkdownContainer"] p { line-height: 1.7; color: #1E293B; }

[data-testid="stMarkdownContainer"] code {
    background: #EEF2FF;
    color: #4361EE;
    padding: 2px 7px;
    border-radius: 6px;
    font-size: 0.87em;
}

[data-testid="stMarkdownContainer"] pre {
    background: #1E293B !important;
    border-radius: 12px !important;
}

/* ---- Follow-up chips (main area secondary buttons) ---- */
.main [data-testid="baseButton-secondary"] {
    background: rgba(238, 242, 255, 0.92) !important;
    color: #4361EE !important;
    border: 1px solid #C7D2FE !important;
    border-radius: 20px !important;
    font-size: 12.5px !important;
    padding: 7px 16px !important;
    font-weight: 500 !important;
    white-space: normal !important;
    height: auto !important;
    line-height: 1.45 !important;
    text-align: center !important;
    transition: background 0.2s, border-color 0.2s, transform 0.15s !important;
}

.main [data-testid="baseButton-secondary"]:hover {
    background: #C7D2FE !important;
    border-color: #818CF8 !important;
    color: #3451D1 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 3px 12px rgba(67, 97, 238, 0.18) !important;
}

/* ---- Quiz container ---- */
.quiz-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 20px;
    padding-bottom: 16px;
    border-bottom: 1px solid #E2E8F0;
}

.quiz-badge {
    background: linear-gradient(135deg, #4361EE, #7B2FBE);
    color: white;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
}

/* ---- Follow-up label ---- */
.followup-label {
    color: #94A3B8;
    font-size: 10.5px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin: 4px 0 10px 0;
    display: block;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# NEURAL NETWORK SVG — com animações CSS
# =========================================================

_NEURAL_SVG = """<svg viewBox="0 0 320 130" xmlns="http://www.w3.org/2000/svg" style="width:310px;height:125px">
  <defs>
    <filter id="gw">
      <feGaussianBlur stdDeviation="3.5" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <style>
      @keyframes neuron-pulse {
        0%,100%{transform:scale(1);opacity:1}
        50%{transform:scale(1.28);opacity:0.72}
      }
      @keyframes conn-flow {
        from{stroke-dashoffset:18}
        to{stroke-dashoffset:0}
      }
      .an{transform-box:fill-box;transform-origin:center;animation:neuron-pulse 1.6s ease-in-out infinite}
      .ac{stroke-dasharray:12 6;animation:conn-flow 0.85s linear infinite}
    </style>
  </defs>
  <line x1="38" y1="22" x2="120" y2="12" stroke="rgba(255,255,255,0.14)" stroke-width="1.2"/>
  <line x1="38" y1="22" x2="120" y2="45" stroke="rgba(255,255,255,0.14)" stroke-width="1.2"/>
  <line x1="38" y1="22" x2="120" y2="78" stroke="rgba(255,255,255,0.14)" stroke-width="1.2"/>
  <line x1="38" y1="22" x2="120" y2="111" stroke="rgba(255,255,255,0.14)" stroke-width="1.2"/>
  <line x1="38" y1="70" x2="120" y2="12" stroke="rgba(255,255,255,0.14)" stroke-width="1.2"/>
  <line x1="38" y1="70" x2="120" y2="78" stroke="rgba(255,255,255,0.14)" stroke-width="1.2"/>
  <line x1="38" y1="70" x2="120" y2="111" stroke="rgba(255,255,255,0.14)" stroke-width="1.2"/>
  <line x1="38" y1="118" x2="120" y2="12" stroke="rgba(255,255,255,0.14)" stroke-width="1.2"/>
  <line x1="38" y1="118" x2="120" y2="45" stroke="rgba(255,255,255,0.14)" stroke-width="1.2"/>
  <line x1="38" y1="118" x2="120" y2="78" stroke="rgba(255,255,255,0.14)" stroke-width="1.2"/>
  <line x1="38" y1="118" x2="120" y2="111" stroke="rgba(255,255,255,0.14)" stroke-width="1.2"/>
  <line x1="120" y1="12" x2="205" y2="12" stroke="rgba(255,255,255,0.14)" stroke-width="1.2"/>
  <line x1="120" y1="12" x2="205" y2="45" stroke="rgba(255,255,255,0.14)" stroke-width="1.2"/>
  <line x1="120" y1="12" x2="205" y2="78" stroke="rgba(255,255,255,0.14)" stroke-width="1.2"/>
  <line x1="120" y1="12" x2="205" y2="111" stroke="rgba(255,255,255,0.14)" stroke-width="1.2"/>
  <line x1="120" y1="45" x2="205" y2="12" stroke="rgba(255,255,255,0.14)" stroke-width="1.2"/>
  <line x1="120" y1="45" x2="205" y2="45" stroke="rgba(255,255,255,0.14)" stroke-width="1.2"/>
  <line x1="120" y1="45" x2="205" y2="111" stroke="rgba(255,255,255,0.14)" stroke-width="1.2"/>
  <line x1="120" y1="78" x2="205" y2="12" stroke="rgba(255,255,255,0.14)" stroke-width="1.2"/>
  <line x1="120" y1="78" x2="205" y2="45" stroke="rgba(255,255,255,0.14)" stroke-width="1.2"/>
  <line x1="120" y1="78" x2="205" y2="78" stroke="rgba(255,255,255,0.14)" stroke-width="1.2"/>
  <line x1="120" y1="78" x2="205" y2="111" stroke="rgba(255,255,255,0.14)" stroke-width="1.2"/>
  <line x1="120" y1="111" x2="205" y2="12" stroke="rgba(255,255,255,0.14)" stroke-width="1.2"/>
  <line x1="120" y1="111" x2="205" y2="45" stroke="rgba(255,255,255,0.14)" stroke-width="1.2"/>
  <line x1="120" y1="111" x2="205" y2="78" stroke="rgba(255,255,255,0.14)" stroke-width="1.2"/>
  <line x1="120" y1="111" x2="205" y2="111" stroke="rgba(255,255,255,0.14)" stroke-width="1.2"/>
  <line x1="205" y1="12" x2="285" y2="22" stroke="rgba(255,255,255,0.14)" stroke-width="1.2"/>
  <line x1="205" y1="12" x2="285" y2="70" stroke="rgba(255,255,255,0.14)" stroke-width="1.2"/>
  <line x1="205" y1="12" x2="285" y2="118" stroke="rgba(255,255,255,0.14)" stroke-width="1.2"/>
  <line x1="205" y1="45" x2="285" y2="22" stroke="rgba(255,255,255,0.14)" stroke-width="1.2"/>
  <line x1="205" y1="45" x2="285" y2="70" stroke="rgba(255,255,255,0.14)" stroke-width="1.2"/>
  <line x1="205" y1="45" x2="285" y2="118" stroke="rgba(255,255,255,0.14)" stroke-width="1.2"/>
  <line x1="205" y1="78" x2="285" y2="22" stroke="rgba(255,255,255,0.14)" stroke-width="1.2"/>
  <line x1="205" y1="78" x2="285" y2="118" stroke="rgba(255,255,255,0.14)" stroke-width="1.2"/>
  <line x1="205" y1="111" x2="285" y2="22" stroke="rgba(255,255,255,0.14)" stroke-width="1.2"/>
  <line x1="205" y1="111" x2="285" y2="70" stroke="rgba(255,255,255,0.14)" stroke-width="1.2"/>
  <line x1="205" y1="111" x2="285" y2="118" stroke="rgba(255,255,255,0.14)" stroke-width="1.2"/>
  <line class="ac" x1="38" y1="70" x2="120" y2="45" stroke="rgba(255,255,255,0.9)" stroke-width="2.5"/>
  <line class="ac" x1="120" y1="45" x2="205" y2="78" stroke="rgba(255,255,255,0.9)" stroke-width="2.5"/>
  <line class="ac" x1="205" y1="78" x2="285" y2="70" stroke="rgba(255,255,255,0.9)" stroke-width="2.5"/>
  <circle cx="38" cy="22" r="9" fill="rgba(255,255,255,0.55)"/>
  <circle class="an" cx="38" cy="70" r="11" fill="white" filter="url(#gw)"/>
  <circle cx="38" cy="118" r="9" fill="rgba(255,255,255,0.55)"/>
  <circle cx="120" cy="12" r="8" fill="rgba(255,255,255,0.45)"/>
  <circle class="an" cx="120" cy="45" r="10" fill="white" filter="url(#gw)"/>
  <circle cx="120" cy="78" r="8" fill="rgba(255,255,255,0.45)"/>
  <circle cx="120" cy="111" r="8" fill="rgba(255,255,255,0.45)"/>
  <circle cx="205" cy="12" r="8" fill="rgba(255,255,255,0.45)"/>
  <circle cx="205" cy="45" r="8" fill="rgba(255,255,255,0.45)"/>
  <circle class="an" cx="205" cy="78" r="10" fill="white" filter="url(#gw)"/>
  <circle cx="205" cy="111" r="8" fill="rgba(255,255,255,0.45)"/>
  <circle cx="285" cy="22" r="9" fill="rgba(255,255,255,0.55)"/>
  <circle class="an" cx="285" cy="70" r="11" fill="white" filter="url(#gw)"/>
  <circle cx="285" cy="118" r="9" fill="rgba(255,255,255,0.55)"/>
</svg>"""

# =========================================================
# HEADER
# =========================================================

st.markdown(f"""
<div class="neural-header">
  <div class="header-left">
    <span class="header-badge">AI Professor</span>
    <h1 class="header-title">🧠 Neural Prof</h1>
    <p class="header-subtitle">Seu assistente especializado em Deep Learning e Inteligência Artificial. Pergunte sobre redes neurais, modelos e frameworks.</p>
    <div class="header-tags">
      <span class="header-tag">Redes Neurais</span>
      <span class="header-tag">Transformers</span>
      <span class="header-tag">LLMs</span>
      <span class="header-tag">PyTorch</span>
      <span class="header-tag">Foundation Models</span>
    </div>
  </div>
  <div class="header-right">{_NEURAL_SVG}</div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("""
    <div class="sidebar-brand">
      <span class="sb-icon">🧠</span>
      <h2>Neural Prof</h2>
      <p>Assistente de Deep Learning</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<span class="topics-label">Nível do aluno</span>', unsafe_allow_html=True)
    st.segmented_control(
        "Nível",
        options=list(LEVELS.keys()),
        default="Intermediário",
        key="difficulty_level",
        label_visibility="collapsed",
    )

    st.markdown("---")

    st.markdown('<span class="topics-label">Sugestões de perguntas</span>', unsafe_allow_html=True)

    for i, topic in enumerate(TOPICS):
        if st.button(topic["label"], key=f"topic_{i}", use_container_width=True):
            st.session_state.prefill_input = topic["question"]
            st.rerun()

    st.markdown("---")

    if st.button("🗑️  Limpar conversa", key="btn_limpar",
                 type="primary", use_container_width=True):

        logger.warning(
            "session_id=%s conversation cleared by user",
            st.session_state.session_id,
        )

        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": "Olá! Como posso ajudar você com Deep Learning?"
            }
        ]
        st.session_state.pop("followup_questions", None)
        st.session_state.pop("quiz_data", None)
        st.session_state.pop("quiz_submitted", None)
        st.session_state.pop("quiz_loading", None)

        st.rerun()

    # Quiz button — aparece só quando há histórico de conversa
    if len(st.session_state.get("chat_history", [])) > 1:
        st.markdown("---")
        st.markdown('<span class="topics-label">Avaliação</span>', unsafe_allow_html=True)
        if st.button("📝  Gerar Quiz", key="btn_quiz", use_container_width=True):
            st.session_state.quiz_loading = True
            st.session_state.pop("quiz_data", None)
            st.session_state.pop("quiz_submitted", None)
            for i in range(3):
                st.session_state.pop(f"quiz_q_{i}", None)
            st.rerun()

# =========================================================
# HISTÓRICO
# =========================================================

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    logger.info("session_id=%s session started", st.session_state.session_id)

if "chat_history" not in st.session_state:

    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": "Olá! Como posso ajudar você com Deep Learning?"
        }
    ]

# =========================================================
# PRÉ-PREENCHIMENTO DO CHAT INPUT VIA JS
# =========================================================

if "prefill_input" in st.session_state:
    _question = st.session_state.pop("prefill_input")
    components.html(
        f"""<script>
        (function() {{
            var text = {json.dumps(_question)};
            function fill() {{
                var ta = window.parent.document.querySelector(
                    '[data-testid="stChatInput"] textarea'
                );
                if (!ta) {{ setTimeout(fill, 200); return; }}
                var setter = Object.getOwnPropertyDescriptor(
                    window.HTMLTextAreaElement.prototype, 'value'
                ).set;
                setter.call(ta, text);
                ta.dispatchEvent(new Event('input', {{ bubbles: true }}));
                ta.focus();
            }}
            setTimeout(fill, 150);
        }})();
        </script>""",
        height=0,
    )

# =========================================================
# EXIBE HISTÓRICO
# =========================================================

for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# =========================================================
# FOLLOW-UP CHIPS — exibidos após última resposta do assistente
# =========================================================

followup_qs = st.session_state.get("followup_questions", [])
quiz_active = bool(st.session_state.get("quiz_data") or st.session_state.get("quiz_loading"))

if followup_qs and not quiz_active:
    st.markdown('<span class="followup-label">Perguntas relacionadas</span>', unsafe_allow_html=True)
    cols = st.columns(len(followup_qs))
    for i, (col, q) in enumerate(zip(cols, followup_qs)):
        with col:
            if st.button(q, key=f"followup_{i}", use_container_width=True):
                st.session_state.pop("followup_questions", None)
                st.session_state.prefill_input = q
                st.rerun()

# =========================================================
# QUIZ — geração e exibição
# =========================================================

if st.session_state.get("quiz_loading"):
    with st.spinner("🎯 Gerando quiz personalizado..."):
        quiz = _generate_quiz(st.session_state.chat_history)
    st.session_state.pop("quiz_loading")
    if quiz:
        st.session_state.quiz_data = quiz
    else:
        st.warning("Não foi possível gerar o quiz. Tente novamente após fazer mais perguntas.")
    st.rerun()

if st.session_state.get("quiz_data"):
    quiz = st.session_state.quiz_data
    questions = quiz.get("questions", [])
    topic = quiz.get("topic", "Deep Learning")
    submitted = st.session_state.get("quiz_submitted", False)

    st.markdown("---")
    st.markdown(f"""
    <div class="quiz-header">
      <span class="quiz-badge">Quiz</span>
      <strong style="font-size:1.05rem;color:#1E293B">📝 {topic}</strong>
    </div>
    """, unsafe_allow_html=True)

    if not submitted:
        for i, q in enumerate(questions):
            st.markdown(f"**{i + 1}. {q['question']}**")
            st.radio(
                f"q{i}",
                q["alternatives"],
                key=f"quiz_q_{i}",
                label_visibility="collapsed",
                index=None,
            )
            st.markdown("")

        col_submit, col_close, _ = st.columns([2, 2, 5])
        with col_submit:
            if st.button("✅ Verificar respostas", key="quiz_submit", type="primary"):
                st.session_state.quiz_submitted = True
                st.rerun()
        with col_close:
            if st.button("✖ Fechar quiz", key="quiz_close"):
                st.session_state.pop("quiz_data", None)
                st.session_state.pop("quiz_submitted", None)
                st.rerun()

    else:
        score = 0
        for i, q in enumerate(questions):
            user_ans = st.session_state.get(f"quiz_q_{i}", "") or ""
            correct_letter = q.get("correct", "A")
            is_correct = bool(user_ans) and user_ans.startswith(f"{correct_letter})")
            if is_correct:
                score += 1

            if is_correct:
                st.success(f"**{i + 1}. {q['question']}** ✅")
            else:
                st.error(f"**{i + 1}. {q['question']}** ❌")
                if user_ans:
                    st.markdown(f"Sua resposta: **{user_ans}**")
                correct_text = next(
                    (a for a in q["alternatives"] if a.startswith(f"{correct_letter})")),
                    f"{correct_letter})"
                )
                st.markdown(f"Resposta correta: **{correct_text}**")

            if q.get("explanation"):
                st.caption(f"💡 {q['explanation']}")
            st.markdown("")

        pct = int((score / len(questions)) * 100) if questions else 0
        if pct == 100:
            st.success(f"### 🏆 Resultado: {score}/{len(questions)} ({pct}%) — Perfeito!")
        elif pct >= 66:
            st.info(f"### 👍 Resultado: {score}/{len(questions)} ({pct}%) — Muito bem!")
        else:
            st.warning(f"### 📚 Resultado: {score}/{len(questions)} ({pct}%) — Continue estudando!")

        col_new, col_close2, _ = st.columns([2, 2, 5])
        with col_new:
            if st.button("🔄 Novo Quiz", key="quiz_new", type="primary"):
                st.session_state.pop("quiz_data", None)
                st.session_state.pop("quiz_submitted", None)
                for j in range(len(questions)):
                    st.session_state.pop(f"quiz_q_{j}", None)
                st.session_state.quiz_loading = True
                st.rerun()
        with col_close2:
            if st.button("✖ Fechar", key="quiz_close2"):
                st.session_state.pop("quiz_data", None)
                st.session_state.pop("quiz_submitted", None)
                for j in range(len(questions)):
                    st.session_state.pop(f"quiz_q_{j}", None)
                st.rerun()

# =========================================================
# INPUT
# =========================================================

user_question = st.chat_input(
    "Digite sua pergunta sobre Deep Learning..."
)

# =========================================================
# PROCESSAMENTO
# =========================================================

if user_question:

    session_id = st.session_state.session_id

    # Limpa chips de follow-up ao iniciar nova pergunta
    st.session_state.pop("followup_questions", None)

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    with st.chat_message("user"):
        st.markdown(user_question)

    logger.info("session_id=%s USER: %s", session_id, user_question)

    _level = st.session_state.get("difficulty_level", "Intermediário") or "Intermediário"
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT + f"\n\n[INSTRUÇÃO DE NÍVEL] {LEVELS[_level]}",
        }
    ]
    for msg in st.session_state.chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    answer = ""

    with st.chat_message("assistant"):
        try:
            def _stream():
                response = openai_client.chat.completions.create(
                    model=_MODEL,
                    messages=messages,
                    stream=True,
                )
                for chunk in response:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta

            answer = st.write_stream(_stream())

        except Exception as error:
            answer = (
                "Desculpe, ocorreu um erro ao processar sua pergunta. "
                "Por favor, tente novamente."
            )
            logger.error(
                "session_id=%s ERROR generating assistant response: %s",
                session_id,
                error,
                exc_info=True,
            )
            st.error(answer)

    logger.info("session_id=%s ASSISTANT: %s", session_id, answer)

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    # Gera sugestões de follow-up (apenas se resposta foi válida)
    if answer and len(answer) > 80:
        with st.spinner("💡 Gerando sugestões..."):
            followups = _get_followup_questions(user_question, answer)
        if followups:
            st.session_state.followup_questions = followups

    st.rerun()
