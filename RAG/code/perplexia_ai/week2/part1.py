"""Part 1 - Web Search implementation using LangGraph.

This implementation focuses on:
- Setting up web search using Tavily
- Processing search results
- Formatting responses with citations
"""

from typing import Dict, List, Optional, TypedDict
import os
from perplexia_ai.core.chat_interface import ChatInterface
from langchain_community.tools import TavilySearchResults
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END, START

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

# TODO: Define state for the application.

# NOTE: The TODOs are only a direction for you to start with.
# You are free to change the structure of the code as you see fit.
class WebSearchState(TypedDict):
    query: str
    search_results: list[dict[str, str]]
    answer: str

class WebSearchChat(ChatInterface):
    """Week 2 Part 1 implementation for web search using LangGraph."""
    
    def __init__(self):
        self.llm = None
        self.search_tool = None
        self.graph = None

    def web_search(self, state: WebSearchState):
        results = self.search_tool.invoke({"query": state["query"]})
        return {"search_results": results}
    
    def summarize_results(self, state: WebSearchState):
        chain= WEB_SEARCH_SUMMARIZE_PROMPT | self.llm | StrOutputParser()
        answer = chain.invoke({"query": state["query"], "search_results": state["search_results"]})
        return {"answer": answer}
    
    
    def initialize(self) -> None:
        """Initialize components for web search.
        
        Students should:
        - Initialize the LLM
        - Set up Tavily search tool
        - Create a LangGraph for web search workflow
        """
        # Set up API key for Tavily
        # os.environ["TAVILY_API_KEY"] = "your-tavily-api-key"
        
        # Initialize LLM
        self.llm = init_chat_model("gpt-4o-mini", model_provider="openai")
        
        # Initialize search tool
        self.search_tool = TavilySearchResults(
            max_results=5,  # Reduced from 5 to 3 to limit content
            include_answer=False,
            include_raw_content=False,  # Changed to False to reduce token count
            include_images=False,
            search_depth="advanced",
            api_key=os.environ["TAVILY_API_KEY"]
        )
        # TODO: Create the graph
        
        # Build the graph
        graph = StateGraph(WebSearchState)
        graph.add_node("web_search", self.web_search)
        graph.add_node("summarize_results", self.summarize_results)
        
        # Define the edges and the graph structure
        graph.add_edge(START, "web_search")
        graph.add_edge("web_search", "summarize_results")
        graph.add_edge("summarize_results", END)
        
        # Compile the graph
        self.graph = graph.compile()

    
    
    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message using web search.
        
        Args:
            message: The user's input message
            chat_history: Previous conversation history
        
        Returns:
            str: The assistant's response with search results
        """
        # 1. Format the input message (if needed)
        # 2. Run the graph
        state={"query": message}
        result = self.graph.invoke(state)
        # 3. Extract the response
        
        return result["answer"] 