import json
from langchain.prompts import ChatPromptTemplate
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI

class RAGGenerator:
    def __init__(self, docs, openai_api_key, memory_file='conversation_memory.json'):
        self.docs = docs
        self.openai_api_key = openai_api_key
        self.memory_file = memory_file
        # model_name=model_name
        self.llm = OpenAI(api_key=openai_api_key, temperature=0)
        self.memory = ConversationBufferMemory(max_size=2000) 
        self.load_memory()

    
    def generate_response(self, question):
        llm =  ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo")
        retrieved_docs = self.docs
        context = "\n\n".join([doc[0].page_content for doc in retrieved_docs])

        ANSWER_PROMPT = ChatPromptTemplate.from_template(
            """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Be as verbose and educational in your response as possible.

            context: {context}
            Question: "{question}"
            Answer:
            """
        )

        chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | ANSWER_PROMPT
            | llm
            | StrOutputParser()
        )

        ans = chain.invoke({"context":context, "question":question})
        return ans
    
    def generate_summary(self):
        llm =  ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo")
        retrieved_docs = self.docs
        # context = "\n\n".join([doc[0].page_content for doc in retrieved_docs])

        ANSWER_PROMPT = ChatPromptTemplate.from_template(
            """You are an assistant for generating summary based on the ccontent user gave you.

            context: {context}
            Answer:
            """
        )

        chain = (
            {"context": RunnablePassthrough()}
            | ANSWER_PROMPT
            | llm
            | StrOutputParser()
        )

        ans = chain.invoke({"context":retrieved_docs})
        return ans

    def save_memory(self):
        """
        Save the conversation buffer memory to a JSON file.
        """
        with open(self.memory_file, 'w') as file:
            json.dump(self.memory.load_memory(), file)

    def load_memory(self):
        """
        Load the conversation buffer memory from a JSON file.
        """
        try:
            with open(self.memory_file, 'r') as file:
                memory_data = json.load(file)
                self.memory.load_memory(memory_data)
        except (FileNotFoundError, json.JSONDecodeError):
            self.memory.clear()

# if __name__ == "__main__":
    
    # rag_generator = RAGGenerator(index_file, docs_folder, openai_api_key)
    
    # question = "What are the recent advancements in renewable energy?"
    
    # response = rag_generator.generate_response(question)
    # print("Assistant's Answer:", response)
