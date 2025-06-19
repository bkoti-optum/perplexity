"""Part 3 - Corrective RAG-lite implementation using LangGraph.

This implementation focuses on:
- Intelligent routing between document knowledge and web search
- Relevance assessment of document chunks
- Combining multiple knowledge sources
- Handling information conflicts
"""

from typing import Dict, List, Optional, Any
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
from langchain.schema import Document
# NOTE: The TODOs are only a direction for you to start with.
# You are free to change the structure of the code as you see fit.

WEB_SEARCH_SUMMARIZE_PROMPT = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant that summarizes web search results to answer the user's query.
    UserQuery: {query}
    Search Results: {search_results}

    Provide the answer in the following format:
    Answer: <answer>
    References:
      - <reference1>
      - <reference2>
      ...
      

    where each reference is a url from the search results.
    
    """
)

class IngestState(TypedDict):
    query: str
    documents: list
    relevance: bool
    search_results: list[dict[str, str]]
    answer: str

class CorrectiveRAGChat(ChatInterface):
    """Week 2 Part 3 implementation for Corrective RAG."""
    
    def __init__(self):
        self.llm = None
        self.embeddings = None
        self.vector_store = None
        self.search_tool = None
        self.document_paths = []
        self.graph = None
    
    def initialize(self) -> None:
        """Initialize components for Corrective RAG.
        
        Students should:
        - Initialize the LLM
        - Set up document loading and processing
        - Create vector embeddings
        - Set up Tavily search tool
        - Build a Corrective RAG workflow using LangGraph
        """
        # TODO: Initialize LLM
        self.llm = init_chat_model("gpt-4o-mini", model_provider="openai")
        
        # TODO: Initialize embeddings
        file_path = glob.glob("/Users/bindukoti/Downloads/RAGDataset/*.pdf")
        pages = []
        for file in file_path:
            loader = PyPDFLoader(file)
            for page in loader.lazy_load():
                pages.append(page)
                
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = []
        for page in pages:
            page_chunks = splitter.split_text(page.page_content)
            chunks.extend(page_chunks)
            
        self.embeddings = OpenAIEmbeddings()
        documents = [Document(page_content=chunk) for chunk in chunks]
        self.vector_store = InMemoryVectorStore.from_documents(documents, self.embeddings)
        
        # TODO: Set up Tavily search tool
        self.search_tool = TavilySearchResults(
            max_results=5,  # Reduced from 5 to 3 to limit content
            include_answer=False,
            include_raw_content=False,  # Changed to False to reduce token count
            include_images=False,
            search_depth="advanced",
            api_key=os.environ["TAVILY_API_KEY"]
        )
        
        # TODO: Set paths to OPM documents
        data_dir = Path("/Users/bindukoti/Downloads/RAGDataset/*.pdf")
        self.document_paths = list(data_dir.glob("*.pdf"))
        # data_dir = Path("path/to/opm/documents")
        # self.document_paths = list(data_dir.glob("*.pdf"))
        
        # TODO: Process documents and create vector store
        # self.vector_store = InMemoryVectorStore.from_documents(self.embeddings, OpenAIEmbeddings())
        # docs = self._load_and_process_documents()
        # self.vector_store = InMemoryVectorStore.from_documents(docs, self.embeddings)
        
        # TODO: Create the graph
        # Define nodes:
        # 1. Document retrieval node: Finds relevant document sections
        # 2. Relevance assessment node: Determines if retrieved documents are relevant
        # 3. Web search node: Performs web search if needed
        
        # Define the graph structure with conditional edges
        def retrieve_documents(state: IngestState):
            docs = self.vector_store.similarity_search(state["query"], k=5)
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
        
        def web_search(state: IngestState):
            results = self.search_tool.invoke({"query": state["query"]})
            return {"search_results": results}
        
        def summarize_results(state: IngestState):
            chain= WEB_SEARCH_SUMMARIZE_PROMPT | self.llm | StrOutputParser()
            answer = chain.invoke({"query": state["query"], "search_results": state["search_results"]})
            return {"answer": answer}
        
        def create_relevance_assessment_score(state: IngestState):
            """Create a node that assesses document relevance."""
            relevance_prompt = ChatPromptTemplate.from_template(
                """Given a query and document content, determine if the document is relevant to answering the query.
                Query: {query}
                Document: {document}
                
                Return only 'true' if the document is relevant, or 'false' if it is not relevant."""
            )
            
            # Check relevance of all documents
            for doc in state["documents"]:
                chain = relevance_prompt | self.llm | StrOutputParser()
                result = chain.invoke({
                    "query": state["query"],
                    "document": doc.page_content
                }).lower().strip()
                if result == "true":
                    return {"relevance": True}
            return {"relevance": False}
        
        def relevance_assessment(state: IngestState):
            if state["relevance"] == True:
                return "generate_answer"
            else:
                return "web_search"
        

        graph = StateGraph(IngestState)

        graph.add_node("retrieve_documents", retrieve_documents)
        graph.add_node("generate_answer", generate_answer)
        graph.add_node("web_search", web_search)
        graph.add_node("summarize_results", summarize_results)
        graph.add_node("create_relevance_assessment_score", create_relevance_assessment_score)
        graph.add_node("relevance_assessment", relevance_assessment)

        graph.add_edge(START, "retrieve_documents")
        graph.add_edge("retrieve_documents", "create_relevance_assessment_score")
        graph.add_conditional_edges(
            "create_relevance_assessment_score",
            relevance_assessment,
            {
                "generate_answer": "generate_answer",
                "web_search": "web_search",
            },
        )
        graph.add_edge("web_search", "summarize_results")
        graph.add_edge("summarize_results", END)
        graph.add_edge("generate_answer", END)

        
        
        # Compile the graph
        self.graph = graph.compile()
        pass
        
        
    
    #def _load_and_process_documents(self) -> list[str]:
        #"""Load and process OPM documents."""
        # TODO: Implement document loading and processing
        # 1. Load the documents
        # 2. Split into chunks
        # 3. Return processed documents
        #return []
    
    #def _create_relevance_assessment_node(self):
        """Create a node that assesses document relevance."""
        # TODO: Implement relevance assessment node
        #pass
    
    #def _create_document_retrieval_node(self):
        """Create a node that retrieves relevant document sections."""
        # TODO: Implement document retrieval node
        #pass
    
    #def _create_web_search_node(self):
        #"""Create a node that performs web search when needed."""
        # TODO: Implement web search node
        #pass
    
    #def _should_use_web_search(self, state: Dict[str, Any]) -> bool:
        """Determine if web search should be used based on document relevance."""
        # TODO: Implement logic to decide when to use web search
        #return False
    
    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message using Corrective RAG.
        
        Intelligently combines document knowledge with web search:
        - Uses documents when they contain relevant information
        - Falls back to web search when documents are insufficient
        - Combines information from both sources when appropriate
        
        Args:
            message: The user's input message
            chat_history: Previous conversation history
            
        Returns:
            str: The assistant's response combining document and web knowledge
        """
        # TODO: Implement Corrective RAG processing
        # 1. Format the input message
        # 2. Run the graph
        # 3. Extract the response
        state={"query": message, "documents": [], "answer": "", "relevance": 0, "search_results": []}
        result = self.graph.invoke(state)
        # This is just a placeholder
        return result["answer"] 
        # This is just a placeholder
        #return f"Corrective RAG result for: {message}" 
