import os
import uuid
import logging
from logging.handlers import RotatingFileHandler

import streamlit as st
from dotenv import load_dotenv

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

chat = ChatOpenAI(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("NVIDIA_API_KEY"),
    base_url="https://integrate.api.nvidia.com/v1"
)

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
# STREAMLIT
# =========================================================

st.set_page_config(
    page_title="Professor de Deep Learning!",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Professor de Deep Learning!")

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("Sobre")

    st.markdown("""
    Assistente especializado em Deep Learning.

    Faça perguntas sobre:
    - **Fundamentos de IA e ML:** Regressão, classificação, funções de custo, otimizadores (SGD, Adam), regularização, backpropagation.
    - **Redes Neurais:** Perceptrons, MLPs, CNNs, RNNs, LSTMs, Autoencoders.
    - **Modelos de Atenção e Transformers:** Arquitetura Transformer, self-attention, codificadores/decodificadores, BERT, GPT, T5.
    - **Modelos Fundacionais (Foundation Models):** LLMs, modelos multimodais, difusão, fine-tuning, RAG, RLHF.
    - **Prática e Ferramentas:** PyTorch, TensorFlow, Keras, Hugging Face, ambientes de treinamento, métricas de avaliação.
    """)

    st.markdown("---")

    if st.button("Limpar conversa"):

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
# EXIBE HISTÓRICO
# =========================================================

for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# =========================================================
# INPUT
# =========================================================

user_question = st.chat_input(
    "Digite sua pergunta..."
)

# =========================================================
# PROCESSAMENTO
# =========================================================

if user_question:

    session_id = st.session_state.session_id

    # Exibe usuário

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    with st.chat_message("user"):

        st.markdown(user_question)

    logger.info("session_id=%s USER: %s", session_id, user_question)

    # Monta contexto

    conversation = SYSTEM_PROMPT + "\n\n"

    for msg in st.session_state.chat_history:

        conversation += (
            f"{msg['role']}: "
            f"{msg['content']}\n"
        )

    # Gera resposta

    with st.chat_message("assistant"):

        with st.spinner("Pensando..."):

            try:
                response = chat.chat(
                    conversation,
                    stream=False
                )

                answer = str(response)
                logger.info("session_id=%s ASSISTANT: %s", session_id, answer)
                st.markdown(answer)

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

    # Salva resposta

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
