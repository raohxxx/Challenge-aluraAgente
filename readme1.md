# Agente RAG Interactivo con LangChain, Cohere y Streamlit

Este proyecto implementa un agente de Generación Aumentada por Recuperación (RAG) utilizando la versión moderna de **LangChain 1.x**, los embeddings y modelos generativos gratuitos (plan *trial*) de **Cohere**, y una base de datos vectorial local con **FAISS**. Todo el sistema está desplegado bajo una interfaz web interactiva con **Streamlit** que simula un entorno de chat continuo con memoria de sesión y sugerencias de preguntas orientadas al concepto **tienda online Bim Bam**. usando la documentacion sugerida en el challenge.

---

## 🚀 Características Principales

*   **Persistencia Vectorial (FAISS):** Los archivos PDF contenidos en el directorio especificado se indexan únicamente la primera vez. El índice se guarda localmente para evitar re-indexar documentos existentes, optimizando el tiempo de carga y el uso de tokens.
*   **Modelo de Lenguaje Actualizado:** Integración con el ecosistema oficial `langchain-cohere` usando el modelo vigente de producción para evitar errores de obsolescencia.
*   **Chat Continuo en Streamlit:** Bucle interactivo gestionado mediante `st.session_state` que mantiene la conversación fluida hasta que el usuario decida finalizarla explícitamente.
*   **Especialización en "Bim Bam":** Prompt del sistema modificado para dar respuestas prioritarias, estructuradas y con ejemplos sobre este concepto.
*   **Preguntas de Ejemplo:** Barra lateral con accesos rápidos para evaluar el comportamiento del agente de forma inmediata.

---

## 🛠️ Arquitectura del Código

El proyecto se divide en dos componentes principales: el motor del RAG (`rag.py`) y la interfaz de usuario (`app.py`).

### 1. Motor RAG (`rag.py`)
Gestiona la carga de documentos mediante `PyPDFDirectoryLoader`, la fragmentación de texto, la generación de embeddings multilingües, la persistencia en FAISS y la construcción de la cadena de ejecución mediante **LCEL** (LangChain Expression Language).

```python
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_cohere import ChatCohere, CohereEmbeddings

load_dotenv()

PDF_DIR = os.getenv("PDF_DIRECTORY", "./documentos")
INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "./faiss_index")

# Prompt adaptado al rol del agente y al concepto Bim Bam
PROMPT = ChatPromptTemplate.from_template("""
Eres un asistente experto que responde de manera precisa basándote en el contexto proveído. 
También tienes conocimiento especializado sobre "bim bam" (un concepto/sistema clave para el usuario).

Si la pregunta es sobre "bim bam", utiliza el contexto y tu conocimiento especializado para dar una respuesta clara, estructurada y con ejemplos prácticos de uso si es necesario.

Si no encuentras la información en el contexto ni se relaciona con "bim bam", di textualmente: "No encontré esa información en los documentos."

Contexto:
{context}

Pregunta:
{question}

Respuesta:
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def load_or_build_vectorstore(pdf_directory: str, embeddings):
    index_path = Path(INDEX_DIR)
    
    # 1. Cargar índice si ya existe localmente
    if index_path.exists() and any(index_path.iterdir()):
        return FAISS.load_local(
            str(index_path), 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    
    # 2. Si no existe, crear el directorio y procesar los PDFs
    pdf_path = Path(pdf_directory)
    if not pdf_path.exists():
        pdf_path.mkdir(parents=True, exist_ok=True)
        raise FileNotFoundError(f"Se ha creado el directorio vacío: {pdf_path.resolve()}. Por favor añade tus PDFs e inicia de nuevo.")
        
    loader = PyPDFDirectoryLoader(str(pdf_path))
    documents = loader.load()
    
    if not documents:
        raise ValueError(f"No hay archivos PDF válidos en: {pdf_path.resolve()}")
        
    # Fragmentación semántica del documento
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    
    # Indexación y persistencia
    vectorstore = FAISS.from_documents(chunks, embeddings)
    index_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_path))
    
    return vectorstore

def build_rag_chain(pdf_directory: str = PDF_DIR):
    # Embeddings v3 Multilingual de Cohere
    embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")
    vectorstore = load_or_build_vectorstore(pdf_directory, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # Modelo vigente con soporte activo para agentes y RAG (Evita errores de obsolescencia)
    llm = ChatCohere(model="command-r-plus-08-2024", temperature=0.2)
    
    # Construcción de la cadena LCEL
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    
    num_pdfs = len(list(Path(pdf_directory).glob("*.pdf")))
    num_chunks = len(vectorstore.index_to_docstore_id)
    
    return chain, num_pdfs, num_chunks