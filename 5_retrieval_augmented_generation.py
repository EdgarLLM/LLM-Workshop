import gradio as gr
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores.chroma import Chroma

from constants import MODEL_PATH, EMBEDDING_MODEL, DB_DIR

import os
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

model_name = EMBEDDING_MODEL
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
embedding_model = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

"""
# List of all supported LLMs: https://python.langchain.com/docs/integrations/llms/
llm = LlamaCpp(
    model_path=MODEL_PATH,
    n_gpu_layers=0,  # Set according to your GPU capabilities
    n_batch=512,
    n_ctx=4096,
    f16_kv=True,  # Set to True for efficiency
    verbose=True,
)
"""

"""# Use OpenAI:

os.environ["OPENAI_API_KEY"] = "your_key"
llm = ChatOpenAI(model_name='gpt-3.5-turbo')
"""

# Use self-hosted model:

llm = ChatOpenAI(
    base_url="http://127.0.0.1:1234/v1",
    openai_api_key="some_dummy_key",
)

# create the chain to answer questions
qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type="stuff",
                                       retriever=retriever,
                                       return_source_documents=True)


def search_document(query):
    # Simulate the function qa_chain, which presumably returns a structure like the one in the image
    similar_docs = qa_chain.invoke(query)

    # Assuming 'result' is a key in the similar_docs dict that contains the LLM output
    llm_result = similar_docs['result']

    # Start the HTML string to store the results and LLM output
    results_html = f"<div><strong><p>{llm_result}</p></strong></div>"

    # Start an unordered list for the sources
    results_html += "<div>Sources:<ul>"

    unique_sources = set()

    # Iterate through the source documents to collect unique sources
    for doc in similar_docs['source_documents']:
        # Extract the source URL
        source_url = doc.metadata['source'] if 'source' in doc.metadata else None
        if source_url:  # Only add the source if the URL is not None
            unique_sources.add(source_url)

    # Add each unique source to the HTML string as a list item
    for source in unique_sources:
        results_html += f"<li><a href='{source}' target='_blank'>{source}</a></li>"

    # Close the unordered list and the container div
    results_html += "</ul></div>"

    return results_html


# Gradio interface setup with HTML output for individual result boxes
iface = gr.Interface(
    fn=search_document,
    inputs=gr.Textbox(label="Enter your question"),
    outputs=gr.HTML(label="Relevant Sections"),  # Using HTML output to format results
    title="Retrieval Augmented Generation",
    description="Enter a question and get a response from an LLM with sources"
)

# Launch the Gradio app
iface.launch()
