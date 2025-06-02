# RAG - ICC (Introdução à Ciência da Computação)

## Instalando as bibliotecas necessárias

- Recomendável criar um ambiente virtual para instalar as bibliotecas (opcional)

- Para instalar as bibliotecas: 

```python3
pip install -r requirements.txt
```

## Passo a Passo

- Primeiramente, popular o banco de dados vetorial com os arquivos presentes na pasta /data

```python3
python3 populate_database.py
```

- Depois, utilizar o script para interagir com o modelo

```python3
python3 rag_query.py "Qual a senha secreta do liamf?"
```

- obs: Importante! Para o modelo funcionar, voce vai precisar do ollama e dos seguintes modelos baixados: "mxbai-embed-large" e "deepseek-r1:7b"

- O servidor ollama precisa estar rodando tambem

```bash
ollama serve
```


