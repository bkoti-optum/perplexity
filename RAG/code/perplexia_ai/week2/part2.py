"""Part 2 - Document RAG implementation using LangGraph.

This implementation focuses on:
- Setting up document loading and processing
- Creating vector embeddings and storage
- Implementing retrieval-augmented generation
- Formatting responses with citations from OPM documents
"""

from perplexia_ai.core.chat_interface import ChatInterface
from typing import Dict, List, Optional, TypedDict
import os
from langchain_community.tools import TavilySearchResults
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END, START
from langchain_community.document_loaders import PyPDFLoader
import glob
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from pathlib import Path



class IngestState(TypedDict):
    query: str
    documents: list
    answer: str

# NOTE: The TODOs are only a direction for you to start with.
# You are free to change the structure of the code as you see fit.
class DocumentRAGChat(ChatInterface):
    """Week 2 Part 2 implementation for document RAG."""
    
    def __init__(self):
        self.llm = None
        self.embeddings = None
        self.vector_store = None
        self.document_paths = []
        self.graph = None
    
    def initialize(self) -> None:
        """Initialize components for document RAG.
        
        Students should:
        - Initialize the LLM
        - Set up document loading and processing
        - Create vector embeddings
        - Build retrieval system
        - Create LangGraph for RAG workflow
        """
        # TODO: Initialize LLM
        self.llm = init_chat_model("gpt-4o-mini", model_provider="openai")

        
        file_path = glob.glob("/Users/bindukoti/Downloads/RAGDataset/*.pdf")
        pages = []
        for file in file_path:
            loader = PyPDFLoader(file)
            for page in loader.lazy_load():
                pages.append(page)


        #print(pages)

        
        # TODO: Initialize embeddings
        
        # TODO: Set paths to OPM documents

        data_dir = Path("/Users/bindukoti/Downloads/RAGDataset/*.pdf")
        self.document_paths = list(data_dir.glob("*.pdf"))

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.embeddings = splitter.split_documents(pages)
        #state["chunks"] = chunks

        
        # TODO: Process documents and create vector store
        self.vector_store = InMemoryVectorStore.from_documents(self.embeddings, OpenAIEmbeddings())
        #docs = self.vector_store.similarity_search("What are OPM's goals?", k=4)
        #print(docs)
        # docs = self._load_and_process_documents()
        # self.vector_store = InMemoryVectorStore.from_documents(docs, self.embeddings)
        
        # TODO: Create the graph
        # Define nodes:
        # 1. Retrieval node: Finds relevant document sections
        # 2. Generation node: Creates response using retrieved context
        
        # Define the edges and the graph structure
        
        # Compile the graph
        

        def retrieve_documents(state: IngestState):
            docs = self.vector_store.similarity_search(state["query"], k=4)
            return {"documents": docs}

        def generate_answer(state: IngestState):
            # Call the model to generate the answer using the retrieved documents
            # and the question
            # Join all the retrieved documents into a single string
            prompt = PromptTemplate.from_template(
              "Answer the question based on the context provided. \n"
              "Question: {question}\n"
              "Context: {context}\n"
              "Answer: "
            )
            context = "\n".join([doc.page_content for doc in state["documents"]])
            prompt = prompt.format(question=state["query"], context=context)
            answer = self.llm.invoke(prompt).content
            return {"answer": answer}
        
        

        graph = StateGraph(IngestState)

        graph.add_node("retrieve_documents", retrieve_documents)
        graph.add_node("generate_answer", generate_answer)

        graph.add_edge(START, "retrieve_documents")
        graph.add_edge("retrieve_documents", "generate_answer")
        graph.add_edge("generate_answer", END)

        
        
       
        self.graph = graph.compile()
        pass
    
    #def _load_and_process_documents(self) -> list[str]:
       # """Load and process OPM documents."""
        # TODO: Implement document loading and processing
        # 1. Load the documents
        # 2. Split into chunks
        # 3. Return processed documents
        #return []
    
    #def _create_retrieval_node(self):
        #"""Create a node that retrieves relevant document sections."""
        # TODO: Implement retrieval node
        #pass
    
    #def _create_generation_node(self):
        # """Create a node that generates responses using retrieved context."""
        # TODO: Implement generation node
        # pass
    
    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message using document RAG.
        
        Should reject queries that are not answerable from the OPM documents.
        
        Args:
            message: The user's input message
            chat_history: Previous conversation history
            
        Returns:
            str: The assistant's response based on document knowledge
        """
        # TODO: Implement document RAG processing
        # 1. Format the input message
        # 2. Run the graph
        # 3. Extract the response
        state={"query": message, "documents": [], "answer": ""}
        result = self.graph.invoke(state)
        # This is just a placeholder
        return result["answer"] 
