import argparse
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Você é um assistente que fala português e é especialista 
da disciplina Introdução à Ciência da Computação.

Responda a pergunta somente com base no seguinte contexto:

{context}

---

Responda a pergunta em português baseado no contexto acima: {question}
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

    results = db.similarity_search_with_score(query_text, k=5)

    context_text = context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    print(prompt)

    model = OllamaLLM(model="deepseek-r1:7b")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()