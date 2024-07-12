import gradio as gr
from data import embedding
from Retriever import retriever
from Generator import generrator
from src import  config
from langchain_core.documents import Document
import os
import shutil


# we wanna use pinecone it's a vector database, elastic search, ranking, and graph.
def chatbot_response(message, file, history=[]):
    print(file)
    # path = "/home/ubuntu/temps/" + os.path.basename(file)  #NB*
    # shutil.copyfile(file.name, path)
    # if path:
    #     try:
    #         with open(file.name, 'rb') as file:
    #             file_content = file.read()
    #             print(file_content)
    #             file_content += f"{file_content}"
    #     except Exception as e:
    #         file_content += f"\n\nError reading file: {str(e)}"
    document = """
    "If we look to the laws, they afford equal justice to all in their private differences...
    if a man is able to serve the state, he is not hindered by the obscurity of his condition. The freedom we enjoy in our government extends also to our ordinary life.
    There, far from exercising adistance_metric jealous surveillance over each other, we do not feel called upon to be angry with our neighbour for doing what he likes..."[15] These lines form the roots of the famous phrase "equal justice under law." The liberality of which Pericles spoke also extended to Athens' foreign policy: "We throw open our city to the world, and never by alien acts exclude foreigners from any opportunity of learning or observing, although the eyes of an enemy may occasionally profit by our liberality..."[16]
    """

    splits = embedding.splitting_text_recursive([Document(page_content=document)])
    #  print(splits)
    vector_chroma = embedding.vector_chroma(splits)
    docs = retriever.retrieve_from_chroma(vector_chroma, query_vector="what is law")
    #  print(docs[0][0].page_content)
    rag_generator = generrator.RAGGenerator(docs, config.OPENAI_API_KEY)
    
    question = "what is law? "
    
    response = rag_generator.generate_response(message)

    print(response)
    history.append(f"User: {message}")
    if "hello" in message.lower():
        response = "Hello! How can I help you today?"
    elif "bye" in message.lower():
        response = "Goodbye! Have a great day!"
    else:
        response = response

    history.append(f"{response}")
    return response


with gr.Blocks(theme="soft") as demo:
    gr.Markdown("<h1 id='header'>Speech Assistant</h1>")
    gr.Markdown("<h3 id='subheader'>Enter your question about speeches and get a comprehensive answer</h3>")

    with gr.Row():
        txt = gr.Textbox(label="Enter your question about speeches", placeholder="Type your question here...", lines=2, elem_id="txt")
    file_upload = gr.File(label="Upload a document", elem_id="file_upload")
    
    btn = gr.Button("Submit")

    with gr.Row():
        outputs = gr.Textbox(label="Generated Response", placeholder="The assistant's response will appear here...", elem_id="outputs")

    btn.click(chatbot_response, inputs=[txt,file_upload], outputs=[outputs])


if __name__ == "__main__":
    demo.launch()
    """_summarpip 
    """