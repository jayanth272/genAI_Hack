from pprint import pprint
from RAG import initialize_vectorstore, create_prompt_templates, create_workflow
import os
import streamlit as st

UPLOAD_FOLDER = '/Users/jajab/Downloads/genAI'  # Define your upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def upload_pdf():
    # file_name = input("Enter the PDF file name (with extension): ")
    file_name = "5sample.json"
    file_path = os.path.join(UPLOAD_FOLDER, file_name)

    # if not os.path.exists(file_path):
    #     print("File does not exist. Please ensure the file is in the specified upload folder.")
    #     return

    print(f"File saved to {file_path}")  # Debug log

    global vectorstore, retriever, question_router, rag_chain, retrieval_grader, hallucination_grader, answer_grader, workflow
    vectorstore = initialize_vectorstore(file_path)
    question_router, rag_chain, retrieval_grader, hallucination_grader, answer_grader = create_prompt_templates()
    workflow = create_workflow(vectorstore, question_router, rag_chain, retrieval_grader, hallucination_grader, answer_grader)

    print("PDF uploaded and vectorstore initialized successfully.")

def chat(prompt):
    global counter
    counter = 0

    # while True:
    #     prompt = input("Enter your question (or type 'exit' to quit): ")
    #     if prompt.lower() == 'exit':
    #         break

    #     result = ""

    #     inputs = {"question": prompt}
    #     for output in workflow.stream(inputs):
    #         for key, value in output.items():
    #             pprint(f"Finished running: {key}:")
    #     pprint(value["generation"])
    #     result = value["generation"]

    #     st.write(result)
    #     print(f"Reply: {result}")

    result = ""
    inputs = {"question": prompt}
    for output in workflow.stream(inputs):
        for key, value in output.items():
            pprint(f"Finished running: {key}:")
    
    pprint(value["generation"])
    result = value["generation"]

    return result

def runChatbot():
    upload_pdf()
    chat()

# if __name__ == '__main__':
    # upload_pdf()  # Upload the PDF and initialize the vectorstore
    # chat()  # Start the chat interface

