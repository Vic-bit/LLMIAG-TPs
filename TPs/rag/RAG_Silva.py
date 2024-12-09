# Librerías
from pinecone import Pinecone, ServerlessSpec
from groq import Groq
import os
from sentence_transformers import SentenceTransformer
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
import time
from tqdm.auto import tqdm
import streamlit as st

# Selección del device
device =  'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"\nDevice: {device}") 

# Cargar las keys del archivo .env
load_dotenv()

#-------------------------------------------------------------------------------------------------- 
# Carga del CV y obtención de los chunks
#-------------------------------------------------------------------------------------------------- 

# Lee documentos
def read_doc(directory):
    """
    Lee el documento PDF.
    """
    file_loader=PyPDFLoader(directory)
    documents = file_loader.load()
    return documents

# Convierte en chunks el documento
def chunk_documents(documents, chunk_size=50, chunk_overlap=15):
    """
    Crea chunks del documento.
    """
    text_splitter = CharacterTextSplitter(
        separator=" ", # Divide por espacios
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = []
    for doc in documents:
        # Divide el contenido del texto en fragmentos
        for chunk in text_splitter.split_text(doc.page_content): 
            chunks.append({
                "text": chunk,
                "metadata": doc.metadata #  # Mantén la fuente original
            })
    return chunks

# Leer el documento
doc = read_doc("../../data/Silva_Victor_CV.pdf")

# Divide los documentos en fragmentos
chunked_docs = chunk_documents(doc)

# Verifica el número de fragmentos generados
print(f"\nTotal de fragmentos generados: {len(chunked_docs)}")

#-------------------------------------------------------------------------------------------------- 
# Carga de vectores a Pinecone
#-------------------------------------------------------------------------------------------------- 

# Procesar documentos y generar embeddings
def process_docs(documents):
    """
    Procesa el documento.
    """
    texts = [doc['text'] for doc in documents]
    embeddings = embed_model.encode(texts, convert_to_tensor=False, show_progress_bar=True)
    return texts, embeddings

# Configurar clave de API de Pinecone
PINECONE_API_KEY = os.getenv("pinecone_api_key")
pc = Pinecone(api_key=PINECONE_API_KEY)
cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'

spec = ServerlessSpec(cloud=cloud, region=region)

# Eliminar índices anteriormente creados
#pc.delete_index("cv")

index_name = 'cv'

# Crea una lista con los índices existentes 
existing_indexes = [
    index_info["name"] for index_info in pc.list_indexes()
]

# check if index already exists (it shouldn't if this is first time)
if index_name not in existing_indexes:
    # if does not exist, create index
    pc.create_index(
        index_name,
        dimension=384,  # dimensionality of minilm
        metric='cosine',
        spec=spec
    )
    # wait for index to be initialized
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

# connect to index
index = pc.Index(index_name)

time.sleep(1)

# view index stats
index.describe_index_stats()

# Configuración del modelo de embedding
embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)

# Obtener el texto y los embeddings de los chunks
texts, embeddings = process_docs(chunked_docs)

# Subir los embeddings a Pinecone
ids = [f"doc_{i}" for i in range(len(texts))]  # Crear IDs únicos para los documentos
vectors = [{"id": doc_id, "values": emb, "metadata": {"text": text}}
           for doc_id, emb, text in zip(ids, embeddings, texts)]

# Subir los vectores al índice en lotes
batch_size = 5
for i in tqdm(range(0, len(vectors), batch_size)):
    batch = vectors[i:i + batch_size]
    index.upsert(batch)

# Ver estadísticas del índice
stats = index.describe_index_stats()

print(f"\nGenerados {len(embeddings)} embeddings.")

#-------------------------------------------------------------------------------------------------- 
# Implementación de un chatbot simple
#-------------------------------------------------------------------------------------------------- 

# Carga la clave de API de GROQ desde las variables de entorno
GROQ_API_KEY = os.getenv("groq_api_key")

# Crea el cliente de GROQ
groq_client = Groq(api_key=GROQ_API_KEY)

def query_pinecone(index, query, top_k=5):
    """
    Consulta Pinecone y devuelve los documentos más relevantes.
    """
    # Crear el vector de la consulta
    query_vector = embed_model.encode(query).tolist()
    
    # Consultar Pinecone
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    return results["matches"]

def create_prompt(query, pinecone_matches):
    """
    Crea un prompt para Groq utilizando la consulta y los documentos relevantes.
    """
    context = "\n".join([match["metadata"]["text"] for match in pinecone_matches])
    prompt = f"""
    Contexto relevante:
    {context}

    Pregunta:
    {query}

    Responde con información relevante al contexto y de forma concisa:
    """
    return prompt

def ask_groq(client, prompt, model="llama3-8b-8192"):
    """
    Envía el prompt a Groq y devuelve la respuesta.
    """
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=0.0
    )
    return response.choices[0].message.content

#-------------------------------------------------------------------------------------------------- 
# Streamlit App
#-------------------------------------------------------------------------------------------------- 

# Inicializa el historial de conversación en el estado de la sesión
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Generar respuesta con historial
def generate_response_with_history(input_text, index, embed_model, groq_client):
    """
    Genera una respuesta del chatbot utilizando historial y contexto relevante de Pinecone.
    """
    # Agrega el mensaje del usuario al historial de conversación
    st.session_state.conversation_history.append({"role": "user", "content": input_text})

    # Consulta Pinecone para obtener el contexto relevante
    matches = query_pinecone(index, input_text, top_k=5)
    
    # Crea el prompt basado en el contexto y el historial
    context = "\n".join([match["metadata"]["text"] for match in matches])
    conversation_history = "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in st.session_state.conversation_history]
    )
    prompt = f"""
    Contexto relevante:
    {context}

    Historial de conversación:
    {conversation_history}

    Responde con información relevante al contexto de forma concisa y clara:
    """

    # Obtén la respuesta del modelo Groq
    response = ask_groq(groq_client, prompt)

    # Agrega la respuesta del asistente al historial de conversación
    st.session_state.conversation_history.append({"role": "assistant", "content": response})

    return response, matches

# Interfaz de Streamlit con historial
st.title("RAG con Pinecone y Groq")
st.subheader("Pregunta sobre el CV de Víctor Silva")

# Input del usuario
user_query = st.text_input("Pregunta:")

if user_query:
    with st.spinner("Buscando respuestas..."):
        # Genera respuesta con historial
        response, matches = generate_response_with_history(user_query, index, embed_model, groq_client)
    
    # Mostrar la respuesta
    st.write("**Respuesta del Chatbot:**")
    st.write(response)

    # Mostrar el historial de la conversación
    st.write("**Historial de conversación:**")
    for msg in st.session_state.conversation_history:
        st.write(f"{msg['role'].capitalize()}: {msg['content']}")

    # Mostrar los contextos relevantes
    st.write("**Contexto relevante:**")
    for match in matches:
        st.write(match["metadata"]["text"])
