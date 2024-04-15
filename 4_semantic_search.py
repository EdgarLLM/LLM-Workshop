import gradio as gr
from FlagEmbedding import FlagReranker
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma

from constants import DB_DIR, EMBEDDING_MODEL

import os
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"


# load embedding model
model_name = EMBEDDING_MODEL
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
embedding_model = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

# load database
vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embedding_model)


def semantic_search(query):
    similar_docs = vectorstore.similarity_search(query, k=10)
    results_html = ''
    for doc in similar_docs:
        source_url = doc.metadata["source"]
        result_html = f'''
        <div style="margin-bottom: 20px; border: 1px solid #eee; padding: 10px;">
            {doc.page_content} <br>
            <a href="{source_url}" target="_blank">Source</a>
        </div>
        '''
        results_html += result_html

    return results_html


def semantic_search_with_reranking(query):
    similar_docs = vectorstore.similarity_search(query, k=3)

    # reranking for improved results
    reranker = FlagReranker('BAAI/bge-reranker-large')
    scores = reranker.compute_score([[query, doc.page_content] for doc in similar_docs])
    doc_scores = zip(similar_docs, scores)
    sorted_docs = sorted(doc_scores, key=lambda x: x[1], reverse=True)

    results_html = ''
    # Build the HTML string with sorted documents
    for doc, score in sorted_docs:
        source_url = doc.metadata["source"]
        result_html = f'''
            <div style="margin-bottom: 20px; border: 1px solid #eee; padding: 10px;">
                {doc.page_content} <br>
                Score: {score} <br>
                <a href="{source_url}" target="_blank">Source</a>
            </div>
            '''
        results_html += result_html

    return results_html


# simple gradio interface
iface = gr.Interface(
    fn=semantic_search,
    inputs=gr.Textbox(label="Enter your keyword or question"),
    outputs=gr.HTML(label="Relevant Sections"),
    title="Semantic Search",
    description="Enter a keyword or question to find relevant sections in the document. Each result is displayed in its own styled box."
)

iface.launch()
