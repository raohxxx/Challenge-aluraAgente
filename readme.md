# Agente RAG Interactivo - Documentación de la Tienda Online Bim Bam

Este proyecto implementa un agente de Generación Aumentada por Recuperación (RAG) utilizando la versión moderna de **LangChain 1.x**, los embeddings y modelos generativos de **Cohere**, y una base de datos vectorial local con **FAISS**. Todo el sistema está desplegado bajo una interfaz web interactiva con **Streamlit** que simula un entorno de chat continuo con memoria de sesión.

---

## 📖 Contexto del Proyecto

Este agente inteligente está especializado en actuar como el asistente experto oficial de la **tienda online Bim Bam**. Su propósito principal es consumir, indexar y procesar la documentación interna, manuales, catálogos y políticas de la tienda para resolver de manera inmediata cualquier consulta operativa o comercial. 

Al interactuar con el usuario, el agente está diseñado para ofrecer respuestas prioritarias, estructuradas y acompañadas de ejemplos prácticos basados estrictamente en el contexto de los documentos cargados sobre el ecosistema Bim Bam.

---

## 🚀 Características Principales

*   **Persistencia Vectorial (FAISS):** Los archivos PDF contenidos en el directorio especificado se indexan únicamente la primera vez. El índice se guarda localmente para evitar re-indexar documentos existentes, optimizando el tiempo de carga y el uso de tokens.
*   **Modelo de Lenguaje Actualizado:** Integración con el ecosistema oficial `langchain-cohere` usando el modelo vigente de producción para evitar errores de obsolescencia (`command-r-plus-08-2024`).
*   **Chat Continuo en Streamlit:** Bucle interactivo gestionado mediante `st.session_state` que mantiene la conversación fluida y con memoria del historial de la sesión.
*   **Especialización en Bim Bam:** Prompt del sistema modificado para dar respuestas estructuradas, con ejemplos prácticos orientados a este negocio.
*   **Preguntas de Ejemplo:** Barra lateral con accesos rápidos para evaluar el comportamiento del agente de forma inmediata.

---

## 🛠️ Arquitectura del Código

El proyecto se divide en dos componentes principales:
1.  **Motor RAG (`rag.py`):** Gestiona la carga de documentos mediante `PyPDFDirectoryLoader`, la fragmentación de texto con `RecursiveCharacterTextSplitter`, la generación de embeddings multilingües (`embed-multilingual-v3.0`), la persistencia en FAISS y la construcción de la cadena mediante *LCEL* (LangChain Expression Language).
2.  **Interfaz de Usuario (`app.py`):** Controla el renderizado de la UI de Streamlit, gestiona la sesión de chat y envía las consultas al motor RAG.

---

## 💻 Guía Paso a Paso para Replicar el Proyecto

Sigue estas instrucciones detalladas para clonar, configurar y ejecutar este agente en tu entorno local de desarrollo:

### 1. Clonar el Repositorio
En primer lugar, descarga una copia local del código fuente ejecutando en tu terminal:
```bash
git clone [https://github.com/raohxxx/Challenge-aluraAgente.git](https://github.com/raohxxx/Challenge-aluraAgente.git)
cd Challenge-aluraAgente

### Crear el entorno virtual
python -m venv venv

# Activar el entorno virtual
# En Windows:
venv\Scripts\activate
# En macOS/Linux:
source venv/bin/activate

### 3. Instalar Dependencias
Instala los paquetes y librerías necesarias especificadas en el archivo de requerimientos:
pip install -r requirements.txt
### 4. Configurar Variables de Entorno
Crea un archivo llamado .env en la raíz del proyecto para almacenar tus configuraciones y llaves de API de forma segura. Debe contener la siguiente estructura:
COHERE_API_KEY=tu_api_key_de_cohere_aqui
PDF_DIRECTORY=./documentos
FAISS_INDEX_DIR=./faiss_index

### 5. Cargar la Documentación de Bim Bam
El sistema busca archivos en formato PDF para construir su base de conocimientos.

Crea una carpeta llamada documentos en la raíz del proyecto (si el script no la ha creado automáticamente).

Deposita dentro de esta carpeta todos los documentos PDF correspondientes a las políticas, guías y manuales de la tienda online Bim Bam.

### 6. Ejecutar la Aplicación
Inicia el servidor local de Streamlit para interactuar con el agente en tu navegador web:
streamlit run app.py
Una vez ejecutado, se abrirá automáticamente una pestaña (por defecto en http://localhost:8501) donde podrás empezar a chatear con tu nuevo asistente experto.