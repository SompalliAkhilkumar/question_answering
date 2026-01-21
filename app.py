# Install required packages
# pip install streamlit transformers torch PyPDF2

import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import PyPDF2

# Load T5 model and tokenizer
model_name = "t5-small"  # you can use t5-base or t5-large for better accuracy
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Streamlit app
st.title("PDF Question Answering with T5")
st.write("Upload a PDF and ask questions about its content!")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

pdf_text = ""
if uploaded_file is not None:
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    for page in pdf_reader.pages:
        pdf_text += page.extract_text()
    st.success("PDF loaded successfully!")

# Ask questions
if pdf_text:
    question = st.text_input("Ask a question about the PDF:")
    
    if question:
        # Prepare input for T5
        input_text = f"question: {question}  context: {pdf_text}"
        input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        
        # Generate answer
        outputs = model.generate(input_ids, max_length=150, num_beams=4, early_stopping=True)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        st.write("**Answer:**")
        st.write(answer)
