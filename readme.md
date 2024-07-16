## Question Answering System for PDF Assistance
This project provides a tool for PDF assistance leveraging advanced Retrieval-Augmented Generation (RAG) and graph-based methodologies. It enables users to upload PDF files and interactively query the content, receive summaries, and find relevant references.

- **Chunking**: Recursive splitting
- **Indexing**: ElasticSearch, Pinecone (Ensemble)
- **Pipeline**: Graph Search using Neo4j; if the node doesn't exist, Ensemble RAG is used
- **Frontend**: Gradio
- **Tools**: Langchain, OpenAI
  
## Features

- Question Answering: Upload PDF files and interactively query their content.
  ! [Question-Answering.png](https://github.com/saharkh99/Speech-master/blob/master/images/photo_2024-07-16_11-04-37.jpg)
- Summary: Receive summaries of PDF content
  ! [Summary.png](https://github.com/saharkh99/Speech-master/blob/master/images/photo_2024-07-16_11-04-41.jpg)
- Looking for similar articles on Web: Find relevant references within the PDF.
  ! [search.png](https://github.com/saharkh99/Speech-master/blob/master/images/photo_2024-07-16_11-04-45.jpg)
- General Information:
  ! [information.png](https://github.com/saharkh99/Speech-master/blob/master/images/photo_2024-07-16_11-04-49.jpg)

  ## How to run:
1.  Clone the repository:
2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Update the `src/config.py` file with your credentials. 
4. Run the following command to start the application:

```bash
python -m src.main
```



  








