import streamlit as st
from rag import build_rag_chain

st.set_page_config(page_title="Agente RAG con Cohere", page_icon="🤖")
st.title("🤖 Agente RAG Interactivo (Cohere + FAISS)")

# Inicializar la cadena una sola vez en cache
@st.cache_resource
def load_chain():
    return build_rag_chain()

try:
    chain, num_pdfs, num_chunks = load_chain()
    st.sidebar.success(f"📚 {num_pdfs} PDFs indexados.")
    st.sidebar.info(f"🧩 {num_chunks} Fragmentos en FAISS.")
except Exception as e:
    st.error(f"Error al inicializar el RAG: {e}")
    st.stop()

# Inicializar historial de chat en la sesión de Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_activo" not in st.session_state:
    st.session_state.chat_activo = True

# Mostrar los mensajes anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Evaluar si el chat sigue activo
if st.session_state.chat_activo:
    # Capturar la entrada del usuario de forma continua
    if question := st.chat_input("¿Cuál es tu pregunta? (Escribe 'salir' para finalizar)"):
        
        # Mostrar el mensaje del usuario de inmediato
        with st.chat_message("user"):
            st.write(question)
        st.session_state.messages.append({"role": "user", "content": question})
        
        # Validar si el usuario quiere terminar
        if question.strip().lower() in ["salir", "exit", "quit"]:
            st.session_state.chat_activo = False
            with st.chat_message("assistant"):
                despedida = "Sesión finalizada. ¡Hasta luego!"
                st.write(despedida)
            st.session_state.messages.append({"role": "assistant", "content": despedida})
            st.rerun()
        else:
            # Invocar la respuesta del agente RAG
            with st.chat_message("assistant"):
                with st.spinner("Pensando..."):
                    try:
                        answer = chain.invoke(question)
                        st.write(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as err:
                        st.error(f"Error al conectar con Cohere: {err}")
else:
    st.warning("La sesión ha terminado porque escribiste 'salir'. Si deseas preguntar de nuevo, haz clic abajo:")
    if st.button("Reiniciar Chat"):
        st.session_state.messages = []
        st.session_state.chat_activo = True
        st.rerun()