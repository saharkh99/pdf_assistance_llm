import gradio as gr
from data import embedding
from Retriever import retriever, graph_search
from Generator import generrator
from src import  config
from langchain_core.documents import Document
import PyPDF2
import fitz  

def print_non_empty_metadata(metadata):
    for key, value in metadata.items():
        if value:  
            print(f"{key}: {value}")

def format_string(data):
    formatted_string = ""
    for item in data:
        formatted_string += item + "\n" + "-" * 80 + "\n"
    return formatted_string

def web_search(file):
    print(file)
    try:
        with open(file, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            info = reader.metadata
            num_pages = len(reader.pages)
            content = ""
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                content += page.extract_text()
            title = info.title
            print(title)
            
    except Exception as e:
        print(f"\n\nError reading file: {str(e)}")    
    rag_generator = generrator.RAGGenerator(content, config.OPENAI_API_KEY)    
    response = rag_generator.search_google(config.SEP_API_KEY)
    return format_string(response)
def generate_summary(file):
    print(file)
    try:
        with open(file, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            content = ""
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                content += page.extract_text()
            print(content)
    except Exception as e:
        print(f"\n\nError reading file: {str(e)}")
    document = """
    "If we look to the laws, they afford equal justice to all in their private differences...
    if a man is able to serve the state, he is not hindered by the obscurity of his condition. The freedom we enjoy in our government extends also to our ordinary life.
    There, far from exercising adistance_metric jealous surveillance over each other, we do not feel called upon to be angry with our neighbour for doing what he likes..."[15] These lines form the roots of the famous phrase "equal justice under law." The liberality of which Pericles spoke also extended to Athens' foreign policy: "We throw open our city to the world, and never by alien acts exclude foreigners from any opportunity of learning or observing, although the eyes of an enemy may occasionally profit by our liberality..."[16]
    """

    rag_generator = generrator.RAGGenerator(content, config.OPENAI_API_KEY)    
    response = rag_generator.generate_summary()
    print(response)
    return response


def extract_pdf_info(pdf_path):
    document = fitz.open(pdf_path)
    metadata = document.metadata
    num_pages = document.page_count
    text = ""
    for key, value in metadata.items():
        if value:  
            text += f" {key}: {value} \n "
    text  = text + f"number of pages: {num_pages}"
    return text

# we wanna use pinecone it's a vector database, elastic search, ranking.
def chatbot_response(message, file, history=[]):
    print(file)
    try:
        with open(file, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            content = ""
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                content += page.extract_text()
            print(content)
    except Exception as e:
        print(f"\n\nError reading file: {str(e)}")
    
    
    splits = embedding.splitting_text_recursive([Document(page_content=content)])
    #  print(splits)
    vector_chroma = embedding.vector_chroma(splits)
    docs = retriever.retrieve_from_chroma(vector_chroma, query_vector="what is law")
    #  print(docs[0][0].page_content)
    rag_generator = generrator.RAGGenerator(docs, config.OPENAI_API_KEY)
        
    response = rag_generator.generate_response(message)
    response2 = graph_search.graph_response(content, message)        
    print(response)
    history.append(f"User: {message}")
    if "hello" in message.lower():
        response = "Hello! How can I help you today?"
    elif "bye" in message.lower():
        response = "Goodbye! Have a great day!"
    else:
        response = response

    history.append(f"{response}")
    if response2['result'] == "I don't know the answer.":
        return response
    else: 
        return f"{response2} . {response}" 


with gr.Blocks(theme="soft") as demo:
    gr.Markdown("<h1 id='header'>PDF Assistant</h1>")
    gr.Markdown("<h3 id='subheader'>PDF assistance. you can ask questions about yor private document</h3>")

    with gr.Row():
        txt = gr.Textbox(label="Enter your question ", placeholder="Type your question here...", lines=2, elem_id="txt")
    file_upload = gr.File(label="Upload a document", elem_id="file_upload")
    
    btn = gr.Button("Submit")
    btn2 = gr.Button("generate summary")
    btn3 = gr.Button("search similar articles on web")
    btn4 = gr.Button("general information")

    with gr.Row():
        outputs = gr.Textbox(label="Generated Response", placeholder="The assistant's response will appear here...", elem_id="outputs")

    btn.click(chatbot_response, inputs=[txt,file_upload], outputs=[outputs])
   
    btn2.click(generate_summary, inputs=[file_upload], outputs=[outputs])
    
    btn3.click(web_search, inputs=[file_upload], outputs=[outputs])

    btn4.click(extract_pdf_info, inputs=[file_upload], outputs=[outputs])

if __name__ == "__main__":
    demo.launch()
    """_summarpip 
    """