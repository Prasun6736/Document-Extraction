import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
import torch
import base64

# Model and tokenizer loading
checkpoint = "LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float32, offload_folder='./offload')

# File loader and preprocessing
def file_preprocessing(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    final_texts = ""
    for text in texts:
        final_texts = final_texts + text.page_content
    return final_texts

# LLM pipeline
def llm_pipeline(filepath):
    pipe_sum = pipeline(
        'summarization',
        model=base_model,
        tokenizer=tokenizer,
        max_length=500,
        min_length=50)
    input_text = file_preprocessing(filepath)
    result = pipe_sum(input_text)
    result = result[0]['summary_text']
    return result

@st.cache_data
# Function to display the PDF of a given file
def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

    st.markdown(pdf_display, unsafe_allow_html=True)

# Streamlit code
st.set_page_config(layout="wide")

# Add custom CSS styles
st.markdown("""
    <style>
        .title {
            font-size: 40px;
            color: #4CAF50;
            font-weight: bold;
        }
        .subheader {
            font-size: 30px;
            color: #2196F3;
            margin-top: 20px;
        }
        .button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 18px;
            font-weight: bold;
        }
        .info {
            background-color: #f1f1f1;
            padding: 15px;
            border-radius: 10px;
        }
        .success {
            background-color: #dff0d8;
            padding: 15px;
            border-radius: 10px;
        }
        .col-wrapper {
            padding-top: 20px;
        }
        .file-upload {
            font-size: 18px;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<p class="title">Document Summarization App</p>', unsafe_allow_html=True)

    st.markdown('<p class="subheader">Upload a PDF document to summarize.</p>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'], label_visibility="collapsed")
    
    if uploaded_file is not None:
        if st.button("Summarize", key="summarize", use_container_width=True):
            col1, col2 = st.columns([2, 3], gap="large")
            filepath = "data/" + uploaded_file.name
            
            with open(filepath, "wb") as temp_file:
                temp_file.write(uploaded_file.read())

            with col1:
                st.markdown('<div class="info">Uploaded File</div>', unsafe_allow_html=True)
                displayPDF(filepath)

            with col2:
                st.markdown('<div class="info">Processing and Summarizing...</div>', unsafe_allow_html=True)
                summary = llm_pipeline(filepath)
                st.markdown('<div class="success">Summarization Complete</div>', unsafe_allow_html=True)
                st.success(summary)

if __name__ == "__main__":
    main()
