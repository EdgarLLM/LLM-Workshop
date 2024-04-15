from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma

from constants import EMBEDDING_MODEL, DB_DIR

# load embedding model
model_name = EMBEDDING_MODEL
model_kwargs = {"device": "cpu"}  # cuda
encode_kwargs = {"normalize_embeddings": True}
embedding_model = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

data = WikipediaLoader("Large Language Model", lang="en", load_max_docs=5, doc_content_chars_max=10000).load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)
vectorstore = Chroma.from_documents(documents=all_splits, embedding=embedding_model, persist_directory=DB_DIR)

# To get all documents in the database
db_results = vectorstore.get()
print(len(db_results['ids']), "\n")

similar_docs = vectorstore.similarity_search_with_score("What is a Large Language Model?", k=3)
for doc, score in similar_docs:
    print("Score:", score, "doc.page_content:", doc.page_content, "\n")
