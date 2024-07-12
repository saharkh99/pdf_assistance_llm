import re
import nltk
nltk.download('punkt') 

def preprocess_document(document):
    """
    Preprocesses the input document by cleaning and standardizing the text.
    
    Args:
        document (str): The input text document to preprocess.
    
    Returns:
        str: The cleaned and processed text.
    """
    cleaned_text = re.sub(r'[^A-Za-z\s,.!?\'\"]+', '', document)
    cleaned_text = cleaned_text.lower()
    cleaned_text = re.sub(r'\[\d+\]', '', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    sentences = nltk.sent_tokenize(cleaned_text)
    processed_text = ' '.join(sentences)

    return processed_text

# Example usage

def process_and_save_file(input_path, output_path):
    """
    Reads text from an input file, processes it, and saves the cleaned text to an output file.

    Args:
        input_path (str): The file path of the input text document.
        output_path (str): The file path where the cleaned text will be saved.
    
    Returns:
        str: The cleaned text if successful, otherwise prints an error message.
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as file:
            document = file.read()
        cleaned_text = preprocess_document(document)
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(cleaned_text)

        return cleaned_text

    except Exception as e:
        print(f"An error occurred: {e}")
        return None    

if __name__ == "__main__":

    document = """
    "If we look to the laws, they afford equal justice to all in their private differences...
    if a man is able to serve the state, he is not hindered by the obscurity of his condition. The freedom we enjoy in our government extends also to our ordinary life.
    There, far from exercising a jealous surveillance over each other, we do not feel called upon to be angry with our neighbour for doing what he likes..."[15] These lines form the roots of the famous phrase "equal justice under law." The liberality of which Pericles spoke also extended to Athens' foreign policy: "We throw open our city to the world, and never by alien acts exclude foreigners from any opportunity of learning or observing, although the eyes of an enemy may occasionally profit by our liberality..."[16]
    """
    processed_text = preprocess_document(document)
    print(processed_text)