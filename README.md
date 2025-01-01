# Document-Extraction

This is a web-based application that allows users to upload PDF documents and receive an automatic summary using a pre-trained language model. The application leverages **Streamlit** for the user interface, **LangChain** for document preprocessing, and **Hugging Face Transformers** for utilizing the **LaMini-Flan-T5-248M** model to generate summaries.

https://github.com/Prasun6736/Document-Extraction/blob/538ed5ca01c51e5c7307918ef638318d2273aa82/Screenshot%202025-01-01%20141908.png

## Features

- **PDF File Upload**: Users can upload a PDF document.
- **Text Extraction**: Extracts text from the uploaded PDF document.
- **Summarization**: Uses the T5-based LaMini-Flan-T5-248M model to summarize the document.
- **Real-time Output**: Displays the summarized text in real-time after uploading the document.

## Tech Stack

- **Streamlit**: Web framework to display the app.
- **Hugging Face Transformers**: Pre-trained language model for text summarization.
- **LangChain**: Document preprocessing and text splitting.
- **PyTorch**: Model backend.
- **SentencePiece**: Tokenizer for handling the model.
- **safetensors**: Safely loads and offloads model weights.
- **accelerate**: Optimizes the model for performance and memory.


