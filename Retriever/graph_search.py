from typing import List, Tuple
from langchain.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from src import config
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts.prompt import PromptTemplate

def graph_response(document, question):

    graph = Neo4jGraph(username= config.NEO4J_USERNAME, url = config.NEO4J_URI, password= config.NEO4J_PASSWORD)
    llm = ChatOpenAI(temperature=0, model_name="gpt-4-turbo")
    llm_transformer = LLMGraphTransformer(llm=llm)
    documents = [Document(page_content=document)]
    graph_documents = llm_transformer.convert_to_graph_documents(documents)
    graph.add_graph_documents(graph_documents)
    graph.refresh_schema()
    print(graph.schema)
    CYPHER_GENERATION_TEMPLATE = """Task:Generate Cypher statement to query a graph database.
    Instructions:
    Use only the provided relationship types and properties in the schema.
    Do not use any other relationship types or properties that are not provided.
    Schema:
    {schema}
    Note: Do not include any explanations or apologies in your responses.
    Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
    Do not include any text except the generated Cypher statement.
    The question is:  
    {question}
    """

    CYPHER_GENERATION_PROMPT = PromptTemplate(
        input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
    )

    chain = GraphCypherQAChain.from_llm(
        ChatOpenAI(temperature=0),
        graph=graph,
        verbose=True,
        cypher_prompt=CYPHER_GENERATION_PROMPT,
    )
    response = chain.invoke({"query": question})
    return response

if __name__ == "__main__":
    document = """
    "If we look to the laws, they afford equal justice to all in their private differences...
    if a man is able to serve the state, he is not hindered by the obscurity of his condition. The freedom we enjoy in our government extends also to our ordinary life.
    There, far from exercising adistance_metric jealous surveillance over each other, we do not feel called upon to be angry with our neighbour for doing what he likes..."[15] These lines form the roots of the famous phrase "equal justice under law." The liberality of which Pericles spoke also extended to Athens' foreign policy: "We throw open our city to the world, and never by alien acts exclude foreigners from any opportunity of learning or observing, although the eyes of an enemy may occasionally profit by our liberality..."[16]
    """
    print(graph_response(document, "who "))
    