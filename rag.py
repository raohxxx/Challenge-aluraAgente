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

# Prompt adaptado al rol del agente
# Modifica el PROMPT en tu archivo rag.py
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

# Asegúrate de que abajo, en build_rag_chain, el modelo esté actualizado:
# llm = ChatCohere(model="command-r-plus-08-2024", temperature=0.2)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def load_or_build_vectorstore(pdf_directory: str, embeddings):
    index_path = Path(INDEX_DIR)
    
    # 1. Si el índice FAISS ya existe, lo cargamos directamente para ahorrar tiempo y tokens
    if index_path.exists() and any(index_path.iterdir()):
        return FAISS.load_local(
            str(index_path), 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    
    # 2. Si no existe, lee el directorio de documentos
    pdf_path = Path(pdf_directory)
    if not pdf_path.exists():
        pdf_path.mkdir(parents=True, exist_ok=True)
        raise FileNotFoundError(f"Se ha creado el directorio vacío: {pdf_path.resolve()}. Por favor añade tus PDFs e inicia de nuevo.")
        
    loader = PyPDFDirectoryLoader(str(pdf_path))
    documents = loader.load()
    
    if not documents:
        raise ValueError(f"No hay archivos PDF válidos en: {pdf_path.resolve()}")
        
    # Fragmentar los documentos
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    
    # Crear e indexar con FAISS
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Guardar localmente
    index_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_path))
    
    return vectorstore

def build_rag_chain(pdf_directory: str = PDF_DIR):
    # Usamos los embeddings oficiales de Cohere (V3 es excelente para RAG)
    embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")
    
    vectorstore = load_or_build_vectorstore(pdf_directory, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # El modelo gratuito por defecto de Cohere (trial key) suele ser 'command-r-plus' o 'command'
    #llm = ChatCohere(model="command-r-plus", temperature=0.2)
    llm = ChatCohere(model="command-a-03-2025", temperature=0.2)
    
    # Cadena LCEL
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    
    num_pdfs = len(list(Path(pdf_directory).glob("*.pdf")))
    num_chunks = len(vectorstore.index_to_docstore_id)
    
    return chain, num_pdfs, num_chunks