import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, JSONLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from dotenv import load_dotenv

load_dotenv()

# Set up the environment variable for Google API key
# Make sure to create a .env file with your GOOGLE_API_KEY
if "GOOGLE_API_KEY" not in os.environ:
    print("Please set the GOOGLE_API_KEY environment variable.")
    exit()

# 1. Load and process the documents
text_loader = TextLoader("dataset.txt", encoding="utf-8")
json_loader = JSONLoader("TechNova_Profile.json", jq_schema=".", text_content=False)

documents = text_loader.load()
documents.extend(json_loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(documents)

# 2. Create the vector store
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()


# 3. Define the LangGraph state
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]


# 4. Define the LangGraph nodes
def retrieve_documents(state):
    print("---RETRIEVING DOCUMENTS---")
    question = state["question"]
    documents = retriever.get_relevant_documents(question)
    return {"documents": documents, "question": question}


def generate_answer(state):
    print("---GENERATING ANSWER---")
    question = state["question"]
    documents = state["documents"]

    prompt = PromptTemplate(
        template="""**You are an expert Q&A assistant for TechNova Solutions Inc.**\n\n**Use the following context to answer the question. If you don't know the answer, just say that you don't know.**\n\n**Context:**
{context}\n\n**Question:**
{question}\n\n**Answer:**
""",
        input_variables=["context", "question"],
    )

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    generation = rag_chain.invoke(question)
    return {"documents": documents, "question": question, "generation": generation}


# 5. Build the graph
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("generate", generate_answer)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

# 6. Run the app
if __name__ == "__main__":
    print("RAG Application for TechNova Solutions Inc.")
    print("----------------------------------------")
    while True:
        user_input = input("Ask a question about TechNova (or type 'exit' to quit): ")
        if user_input.lower() == "exit":
            break
        inputs = {"question": user_input}
        for output in app.stream(inputs):
            for key, value in output.items():
                print(f"Finished node '{key}':")
        print(value["generation"])
