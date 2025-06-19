"""Part 1 - Query Understanding implementation.

This implementation focuses on:
- Classify different types of questions
- Format responses based on query type
- Present information professionally
"""

from typing import Dict, List, Optional
from perplexia_ai.core.chat_interface import ChatInterface
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

class QueryUnderstandingChat(ChatInterface):
    """Week 1 Part 1 implementation focusing on query understanding."""
    
    def __init__(self):
        self.llm = None
        self.query_classifier_prompt = None
        self.response_prompts = {}
    
    def initialize(self) -> None:
        """Initialize components for query understanding.
        
        Students should:
        - Initialize the chat model
        - Set up query classification prompts
        - Set up response formatting prompts
        """
        # TODO: Students implement initialization
        #model = init_chat_model("gpt-4o-mini", model_provider="openai")
        self.llm=init_chat_model("gpt-4o-mini", model_provider="openai")
        self.query_classifier_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a query classifier. Your task is to classify user queries into one of these categories:
            1. FACTUAL - Questions seeking factual information
            2. COMPARATIVE - Questions asking to compare things
            3. ANALYTICAL - Questions seeking analysis or reasoning
            4. DEFINITION - Questions seeking the definition of a term
            
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
            ])
        }


        pass
    
    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message using query understanding.
        
        Students should:
        - Classify the query type
        - Generate appropriate response
        - Format based on query type
        
        Args:
            message: The user's input message
            chat_history: Not used in Part 1
            
        Returns:
            str: The assistant's response
        """
        # Classify the query type
        classification = self.llm.invoke(
            self.query_classifier_prompt.format_messages(message=message)
        ).content.strip().upper()
        
        # Map classification to prompt key
        prompt_key = f"{classification.lower()}_query"
        
        # Get the appropriate response prompt
        if prompt_key in self.response_prompts:
            response_prompt = self.response_prompts[prompt_key]
            # Generate response using the appropriate prompt
            response = self.llm.invoke(
                response_prompt.format_messages(question=message)
            ).content
            return response
        else:
            return "I'm not sure how to handle this type of query. Please try rephrasing your question."