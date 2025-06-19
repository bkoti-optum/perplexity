"""Part 2 - Basic Tools implementation.

This implementation focuses on:
- Detect when calculations are needed
- Use calculator for mathematical operations
- Format calculation results clearly
"""

from typing import Dict, List, Optional
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from perplexia_ai.core.chat_interface import ChatInterface
from perplexia_ai.tools.calculator import Calculator


class BasicToolsChat(ChatInterface):
    """Week 1 Part 2 implementation adding calculator functionality."""

    
    def calculate_answer(self, message: str) -> str:
        """Calculate the answer to the question."""
        print(f"Evaluating expression: {message}")
        return str(Calculator.evaluate_expression(message))
   
    def __init__(self):
        self.llm = None
        self.query_classifier_prompt = None
        self.response_prompts = {}
    
    def initialize(self) -> None:
        """Initialize components for basic tools.
        
        Students should:
        - Initialize the chat model
        - Set up query classification prompts
        - Set up response formatting prompts
        """
        self.llm = init_chat_model("gpt-4o-mini", model_provider="openai")
        self.query_classifier_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a query classifier. Your task is to classify user queries into one of these categories:
            1. FACTUAL - Questions seeking factual information
            2. COMPARATIVE - Questions asking to compare things
            3. ANALYTICAL - Questions seeking analysis or reasoning
            4. DEFINITION - Questions seeking the definition of a term
            5. CALCULATION - Questions seeking calculations
            
            Respond with just the category name."""),
            HumanMessage(content="{message}")
        ])
        self.response_prompts = {         
            "factual_query": ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant that provides information which is factual and it is concise and direct."),
                ("user", "{question}")
            ]),
            "analytical_query": ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant that provides information which is analytical and it should include reasoning steps."),
                ("user", "{question}")
            ]),
            "comparative_query": ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant that provides information which is comparative and it should use structured format."),
                ("user", "{question}")
            ]),
            "definition_query": ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant that provides information which is definition and it should include examaples and use cases."),
                ("user", "{question}")
            ]),
            "calculation_query": ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant that provides information which is calculation and you please dont perform the calculation.You should use the calculator tool to perform the calculation.all expressions  in the format of 1+1, 2*3, 4/2, 5-3, 6^2, 7%3, 8! etc. or % should be done by calculator tool"),
                ("user", "{question}")
            ])
        }
        pass

    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message with calculator support.
        
        Students should:
        - Check if calculation needed
        - Use calculator if needed
        - Otherwise, handle as regular query
        
        Args:
            message: The user's input message
            chat_history: Not used in Part 2
            
        Returns:
            str: The assistant's response
        """
        calculator_tool = self.calculate_answer(message)
        # First classify the query
        classification = self.llm.invoke(
            self.query_classifier_prompt.format_messages(message=message)
        ).content.strip().upper()
        
        # If it's a calculation query, use the calculator directly
        if classification == "CALCULATION":
            try:
                result = calculator_tool
                print(result)
                if isinstance(result, float):
                    return f"The result is: {result}"
                else:
                    return result  # Error message from Calculator
            except Exception as e:
                return f"I encountered an error while performing the calculation: {str(e)}"
        
        # For non-calculation queries, use the appropriate response prompt
        prompt_key = f"{classification.lower()}_query"
        if prompt_key in self.response_prompts:
            response_prompt = self.response_prompts[prompt_key]
            response = self.llm.invoke(
                response_prompt.format_messages(question=message)
            ).content
            return response
        
        return "I'm not sure how to handle this type of query. Please try rephrasing your question."
        