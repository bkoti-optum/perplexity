"""Part 1 - Tool-Using Agent implementation.

This implementation focuses on:
- Converting tools from Assignment 1 to use with LangGraph
- Using the ReAct pattern for autonomous tool selection
- Comparing manual workflow vs agent approaches
"""

from typing import Dict, List, Optional, Any, Annotated
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import TavilySearchResults
from langgraph.prebuilt import ToolNode
from perplexia_ai.core.chat_interface import ChatInterface
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, END, START
from datetime import datetime
import os
import contextlib
import io
from perplexia_ai.tools.calculator import Calculator

class IngestState(MessagesState):
    query: str = ""
    answer: str = ""

class ToolUsingAgentChat(ChatInterface):
    """Week 3 Part 1 implementation focusing on tool-using agents."""
    
    def __init__(self):
        self.llm = None
        self.agent_executor = None
        self.tools = []
    
    def initialize(self) -> None:
        """Initialize components for the tool-using agent.
        
        Students should:
        - Initialize the chat model
        - Define tools for calculator, DateTime, and weather
        - Create the ReAct agent using LangGraph
        """
        # TODO: Initialize your chat model
        self.llm = init_chat_model("gpt-4o", model_provider="openai")
        
        # TODO: Create tools using the tool decorator
        self.tools = self._create_tools()
        
        # TODO: Create the ReAct agent
        self.agent_executor = self._create_agent()
        
        # TODO: Create and compile the graph
    
    def _create_tools(self) -> List[Any]:
        """Create and return the list of tools for the agent.
        
        Students should implement:
        - Calculator tool from Assignment 1
        - [Optional] DateTime tool from Assignment 1
        - Weather tool using Tavily search
        
        Returns:
            List: List of tool objects
        """
        # TODO: Implement calculator tool
        @tool
        def calculator(expression: Annotated[str, "The mathematical expression to evaluate"]) -> str:
            """Evaluate a mathematical expression using basic arithmetic operations (+, -, *, /, %, //).
            Examples: '5 + 3', '10 * (2 + 3)', '15 / 3'
            """
            result = Calculator.evaluate_expression(expression)
            if isinstance(result, str) and result.startswith("Error"):
                raise ValueError(result)
            return str(result)
        
        # TODO: Implement DateTime tool
        @tool
        def execute_datetime_code(code: Annotated[str, "Python code to execute for datetime operations"]) -> str:
            """Execute Python code for datetime operations. The code should use datetime or time modules.
            Examples: 
            - 'print(datetime.datetime.now().strftime("%Y-%m-%d"))'
            - 'print(datetime.datetime.now().year)'
            """
            output_buffer = io.StringIO()
            code = f"import datetime\nimport time\n{code}"
            try:
                with contextlib.redirect_stdout(output_buffer):
                    exec(code)
                return output_buffer.getvalue().strip()
            except Exception as e:
                raise ValueError(f"Error executing datetime code: {str(e)}")
        

        
        # TODO: Implement Weather tool using Tavily
        @tool
        def get_weather(location: Annotated[str, "The location to get weather for (city, country)"]) -> str:
            """Get the current weather for a given location using Tavily search.
            Examples: 'New York, USA', 'London, UK', 'Tokyo, Japan'
            """
            search = TavilySearchResults(max_results=3,api_key=os.environ["TAVILY_API_KEY"])
            query = f"what is the current weather temperature in {location} right now"
            results = search.invoke(query)
            
            if not results:
                return f"Could not find weather information for {location}"
            
            # We are using the first result only but you could also provide a more complex
            # response to the LLM by processing the results if required.
            return results[0].get("content", f"Could not find weather information for {location}")
        #def weather_tool(query: str) -> str:
        #    """Search for weather information using Tavily."""
        #    tavily_tool = TavilySearchResults(
        #        max_results=5,  # Reduced from 5 to 3 to limit content
        #        include_answer=False,
        #        include_raw_content=False,  # Changed to False to reduce token count
        #        include_images=False,
        ##        search_depth="advanced",
        #        api_key=os.environ["TAVILY_API_KEY"]
        #    )
        #    return tavily_tool.invoke(query)

        return [calculator, execute_datetime_code, get_weather]
    
    def _create_agent(self) -> Any:
        """Create and return the ReAct agent executor.
        
        Returns:
            Any: The agent executor graph or callable
        """
        # TODO: Create a ReAct agent with access to tools
        
        agent = create_react_agent(self.llm, tools=self.tools)
        
        # TODO: Set up a StateGraph with the agent

        graph = StateGraph(IngestState)
        graph.add_node("agent", agent)
        graph.add_edge(START, "agent")
        graph.add_edge("agent", END)
        
        # TODO: Define entry point and compile

        return graph.compile()
    
    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message using the tool-using agent.
        
        Students should:
        - Send the message to the agent
        - Extract and return the agent's response
        
        Args:
            message: The user's input message
            chat_history: List of previous chat messages
            
        Returns:
            str: The assistant's response
        """
        # TODO: Prepare input for the agent
        result = self.agent_executor.invoke({"messages": [message], "history": chat_history})
        
        # TODO: Run the agent and return the result
        return result["messages"][-1].content
