from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_elasticsearch import ElasticsearchStore
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
# from langchain.graphs import Neo4jGraph
from langchain_community.embeddings import CohereEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document
import time
from src import utils


def splitting_text_recursive(documents, chunk_size=1000, chunk_overlap=200, add_start_index=True):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=add_start_index
    )
    all_splits = text_splitter.split_documents(documents)

    return all_splits

def splitting_text_semantic(documents):
    text_splitter = SemanticChunker(utils.set_open_ai_embedding())
    semantic_chunks = text_splitter.create_documents([d.page_content for d in documents])
    return semantic_chunks

# def graph_vectorization(documents):
#    graph = Neo4jGraph()
#    llm = ChatOpenAI(temperature=0, model_name="gpt-4-turbo")
#    llm_transformer = LLMGraphTransformer(llm=llm)
#    graph_documents = llm_transformer.convert_to_graph_documents(documents)
#    graph.add_graph_documents(graph_documents)
#    return graph

def vector_chroma(splits):
    vectorstore = Chroma.from_documents(documents=splits,
                                        embedding=utils.set_open_ai_embedding())
    return vectorstore

def elastic_vct_search(splits):
    es_store = ElasticsearchStore(
        es_url="http://localhost:9200",
        index_name="test_index",
        strategy=ElasticsearchStore.SparseVectorRetrievalStrategy(model_id=".elser_model_2"),
    ),
    es_store.add_texts(splits)
    return es_store

def pinecone_vct_db(splits):
    # embedding = CohereEmbeddings()
    embeddings = OpenAIEmbeddings()
    pc = Pinecone(api_key="pinecone_api_key")
    index_name = "speech-index" 
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

    index = pc.Index(index_name)
    docsearch = PineconeVectorStore.from_documents(splits, embeddings, index_name=index)
    return docsearch


if __name__ == "__main__":
     document = """
    "If we look to the laws, they afford equal justice to all in their private differences...
    if a man is able to serve the state, he is not hindered by the obscurity of his condition. The freedom we enjoy in our government extends also to our ordinary life.
    There, far from exercising a jealous surveillance over each other, we do not feel called upon to be angry with our neighbour for doing what he likes..."[15] These lines form the roots of the famous phrase "equal justice under law." The liberality of which Pericles spoke also extended to Athens' foreign policy: "We throw open our city to the world, and never by alien acts exclude foreigners from any opportunity of learning or observing, although the eyes of an enemy may occasionally profit by our liberality..."[16]
    """
     splits = splitting_text_recursive([Document(page_content=document)])
     print(splits)
     print(vector_chroma(splits))

