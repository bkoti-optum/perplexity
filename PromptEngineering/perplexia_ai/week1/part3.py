"""Part 3 - Conversation Memory implementation.

This implementation focuses on:
- Maintain context across messages
- Handle follow-up questions
- Use conversation history in responses
"""

from typing import Dict, List, Optional

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from perplexia_ai.core.chat_interface import ChatInterface
from perplexia_ai.tools.calculator import Calculator

class MemoryChat(ChatInterface):
    """Week 1 Part 3 implementation adding conversation memory."""

    def calculate_answer(self, message: str) -> str:
        """Calculate the answer to the question."""
        print(f"Evaluating expression: {message}")
        return str(Calculator.evaluate_expression(message))
    
    def __init__(self):
        self.llm = None
        self.memory = None
        self.query_classifier_prompt = None
        self.response_prompts = {}
        self.calculator = Calculator()
    
    def initialize(self) -> None:
        """Initialize components for memory-enabled chat.
        
        Students should:
        - Initialize the chat model
        - Set up query classification prompts
        - Set up response formatting prompts
        - Initialize calculator tool
        - Set up conversation memory
        """
        self.llm = init_chat_model("gpt-4o-mini", model_provider="openai")
        self.memory = []  # Initialize empty memory list
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
                ("system", """You are a helpful assistant that provides factual information.
                Previous conversation context:
                {context}
                
                Use this context to provide more relevant and contextual responses.
                Keep your responses concise and direct."""),
                ("user", "{question}")
            ]),
            "analytical_query": ChatPromptTemplate.from_messages([
                ("system", """You are a helpful assistant that provides analytical information.
                Previous conversation context:
                {context}
                
                Use this context to provide more relevant analysis.
                Include clear reasoning steps in your response."""),
                ("user", "{question}")
            ]),
            "comparative_query": ChatPromptTemplate.from_messages([
                ("system", """You are a helpful assistant that provides comparative information.
                Previous conversation context:
                {context}
                
                Use this context to provide more relevant comparisons.
                Structure your response in a clear format."""),
                ("user", "{question}")
            ]),
            "definition_query": ChatPromptTemplate.from_messages([
                ("system", """You are a helpful assistant that provides definitions.
                Previous conversation context:
                {context}
                
                Use this context to provide more relevant definitions.
                Include examples and use cases in your response."""),
                ("user", "{question}")
            ]),
            "calculation_query": ChatPromptTemplate.from_messages([
                ("system", """You are a helpful assistant that handles calculations.
                Previous conversation context:
                {context}
                
                Use this context to understand the calculation request better.
                Extract the calculation expression from the question.
                All expressions should be in the format of 1+1, 2*3, 4/2, 5-3, etc."""),
                ("user", "{question}")
            ])
        }
        pass
    
    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message with memory and tools.
        
        Students should:
        - Use chat history for context
        - Handle follow-up questions
        - Use calculator when needed
        - Format responses appropriately
        
        Args:
            message: The user's input message
            chat_history: List of previous chat messages
            
        Returns:
            str: The assistant's response
        """
        # First classify the query
        classification = self.llm.invoke(
            self.query_classifier_prompt.format_messages(message=message)
        ).content.strip().upper()
        
        # Format chat history if available
        context = ""
        if chat_history and len(chat_history) > 0:
            # Chat history is a list of dictionaries with 'role' and 'content' keys
            context = "\n".join([f"{msg['role']}: {msg['content']}" 
                               for msg in chat_history])
            print(context)
        
        # If it's a calculation query, use the calculator with context
        if classification == "CALCULATION":
            try:
                # Get the calculation expression with context
                calc_expression = self.llm.invoke(
                    self.response_prompts["calculation_query"].format_messages(
                        context=context,
                        question=message
                    )
                ).content.strip()
                
                # Perform the calculation
                result = self.calculate_answer(calc_expression)
                print(result)
                if isinstance(result, float):
                    response = f"The result is: {result}"
                else:
                    response = result  # Error message from Calculator
            except Exception as e:
                response = f"I encountered an error while performing the calculation: {str(e)}"
        else:
            # For non-calculation queries, use the appropriate response prompt
            prompt_key = f"{classification.lower()}_query"
            if prompt_key in self.response_prompts:
                response_prompt = self.response_prompts[prompt_key]
                
                # Generate response with context
                response = self.llm.invoke(
                    response_prompt.format_messages(
                        context=context,
                        question=message
                    )
                ).content
            else:
                response = "I'm not sure how to handle this type of query. Please try rephrasing your question."
        
        # Update memory with the new exchange
        self.memory.append({
            "role": "user",
            "content": message
        })
        self.memory.append({
            "role": "assistant",
            "content": response
        })
        
        return response
