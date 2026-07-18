
#nuevo enlace de modificado
import streamlit as st
from rag import build_rag_chain

st.set_page_config(page_title="Asistente virtual Bim Bam", page_icon="🤖")
st.title("🤖 Asistente virtual Bim Bam ")

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

# --- NUEVA SECCIÓN: EJEMPLOS DE PREGUNTAS ---
st.sidebar.write("---")
st.sidebar.subheader("💡 Preguntas de Ejemplo")
st.sidebar.write("Haz clic en cualquier ejemplo para preguntar automáticamente:")

ejemplos = [
    "¿Qué es bim bam y cuáles son sus componentes principales?",
    "¿Cómo se implementa el concepto de bim bam según los documentos?",
    "Dame un ejemplo práctico de una respuesta usando bim bam",
    "¿Cuál es el resumen del archivo PDF principal?"
]

# Variable para capturar si se hizo clic en un ejemplo
pregunta_ejemplo = None
for i, ejemplo in enumerate(ejemplos):
    if st.sidebar.button(ejemplo, key=f"btn_{i}"):
        pregunta_ejemplo = ejemplo
# --------------------------------------------

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
    # Capturar la entrada (ya sea del input manual o de un botón de ejemplo)
    input_usuario = st.chat_input("¿Cuál es tu pregunta? (Escribe 'salir' para finalizar)")
    
    # Si se usó un botón de ejemplo, reemplaza la variable 'question'
    question = input_usuario or pregunta_ejemplo
    
    if question:
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
                with st.spinner("Pensando en tu respuesta..."):
                    try:
                        answer = chain.invoke(question)
                        st.write(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as err:
                        st.error(f"Error al conectar con Cohere: {err}")
            # Si fue un botón de ejemplo, forzamos un rerun para limpiar el estado del botón
            if pregunta_ejemplo:
                st.rerun()
else:
    st.warning("La sesión ha terminado porque escribiste 'salir'. Si deseas preguntar de nuevo, haz clic abajo:")
    if st.button("Reiniciar Chat"):
        st.session_state.messages = []
        st.session_state.chat_activo = True
        st.rerun()