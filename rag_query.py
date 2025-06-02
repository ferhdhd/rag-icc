import argparse
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

SIMILARITY_THRESHOLD = 0.5

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Você é um assistente de modelo de linguagem de IA especialista em Portugues do Brasil e especialista da disciplina Introdução à Ciência da Computação.

Você vai receber uma série de contextos para conseguir responder a pergunta. 
Se achar que algum contexto não contribui para responder a pergunta, pode ignorá-lo.
Se nenhum contexto ajudar na elaboração da resposta e voce nao tiver certeza dela voce deve admitir que não conseguirá responder.

Seguem os contextos:

{context}

---

Responda a pergunta em português brasileiro baseado no contexto acima: {question}
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="O texto da query.")
    args = parser.parse_args()
    query_text = args.query_text
    rag_query(query_text)

def rag_query(query_text: str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_score(query_text, k=10)
    for doc, score in results:
        print(str(doc) + str(score) + "\n")
    filtered_results = [(doc, score) for doc, score in results if score < SIMILARITY_THRESHOLD]

    if not filtered_results:
        context_text = "Nenhum contexto relevante encontrado."
    else:
        context_text = context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in filtered_results])
    
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    print(prompt)

    model = OllamaLLM(model="deepseek-r1:8b")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in filtered_results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()