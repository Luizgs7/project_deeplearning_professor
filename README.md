# Introdução
- Objetivo da atividade: desenvolver uma aplicação de apoio pedagógico em Deep Learning para a disciplina de Projetos de IA da especialização em IA Generativa.
- Visão geral da solução desenvolvida: uma interface Streamlit que conversa com um modelo de linguagem via API NVIDIA/OpenAI, mantendo histórico de sessão e registrando perguntas e respostas em log.

# Infraestrutura
- Configuração da máquina virtual: ambiente Linux com Python 3.13, dependências instaladas via `requirements.txt` e variáveis de ambiente carregadas por `python-dotenv`.
- Sistema operacional utilizado: Linux.
- Recursos computacionais disponíveis: CPU para execução do front-end Streamlit e conexão de rede para chamadas à API externa; o processamento do modelo é realizado pelo serviço de nuvem da Oracle Academy.

# Modelo Escolhido
- Nome do modelo: `openai/gpt-oss-120b` acessado via `ChatOpenAI` e NVIDIA API.
- Justificativa da escolha: modelo grande e de alto desempenho adequado para respostas técnicas e explicações em Deep Learning, com integração via API já suportada pela biblioteca `chatlas`.
- Principais características: arquitetura de larga escala, capacidade de gerar texto técnico estruturado e suporte a múltiplos tópicos avançados de IA e DL.

# Desenvolvimento
- Arquitetura da aplicação: aplicação web Streamlit que mantém o histórico da conversa em `st.session_state`, constrói o prompt completo com `SYSTEM_PROMPT` e envia o contexto ao modelo para gerar a resposta.
- Bibliotecas utilizadas: `streamlit`, `chatlas`, `openai`, `python-dotenv`.
- Estratégia de gerenciamento de credenciais: uso de arquivo `.env` para armazenar a variável `NVIDIA_API_KEY`; esse arquivo não deve ser versionado e deve estar listado em `.gitignore`.

# Implantação
- Processo de publicação na Oracle Cloud: preparar VM Linux com Python, clonar o repositório, instalar dependências com `pip install -r requirements.txt`, configurar `.env` com a chave de API e rodar o app com `streamlit run src/app.py`.
- Principais desafios encontrados: garantir que a chave de API não vaze no repositório, controlar o tamanho do arquivo de log e manter a sessão do usuário isolada por `session_id`.

# Discussão
- Lições aprendidas: a importância de separar configurações sensíveis do código-fonte (`.env`), a utilidade do Streamlit para criar interfaces rápidas e a necessidade de monitorar o tamanho de logs em aplicativos que serão compartilhados.
- Possíveis melhorias futuras: migrar o log para um formato estruturado como JSONL, adicionar gerenciamento de múltiplos usuários com login, implementar persistência de histórico em banco de dados e suportar controle de versão de prompts para auditoria.

