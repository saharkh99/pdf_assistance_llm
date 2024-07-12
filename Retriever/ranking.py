import langchain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank


def ensembler(pinecone_retriever, elastic_retriever, user_query,k):
    ensemble_retriever = langchain.EnsembleRetriever([pinecone_retriever, elastic_retriever], weights=[0.5, 0.5])
    config = {"configurable": {"search_kwargs_faiss": {"k": k}}}
    docs = ensemble_retriever.invoke(user_query, config=config)
    return docs

def flashrank(retriever,query):
   compressor = FlashrankRerank()
   compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )

   compressed_docs = compression_retriever.invoke(
        query
    )
   
   return compressed_docs



    