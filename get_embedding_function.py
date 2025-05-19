from langchain_ollama import OllamaEmbeddings

# define a funcao de embedding para os documentos
def get_embedding_function():
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    return embeddings