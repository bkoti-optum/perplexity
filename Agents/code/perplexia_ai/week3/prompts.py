from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

DOCUMENT_EVALUATOR_PROMPT = PromptTemplate.from_template(
    """
    You are an evaluator that assesses whether retrieved documents provide sufficient information to answer a user's question.
    
    Question: {question}
    Retrieved Documents: {retrieved_docs}
    
    Evaluate whether the documents contain sufficient information to answer the question.
    Consider relevance, completeness, and accuracy of the information.
    
    Return your evaluation in the following format:
    - is_sufficient: true/false
    - feedback: detailed explanation of what's sufficient or what's missing
    """
)

DOCUMENT_SYNTHESIZER_PROMPT = PromptTemplate.from_template(
    """
    You are a synthesizer that combines information from retrieved documents to answer a user's question.
    
    Question: {question}
    Retrieved Documents: {retrieved_docs}
    
    Synthesize a comprehensive answer based on the retrieved documents.
    If the documents don't contain sufficient information, clearly state what's missing.
    
    Provide a well-structured answer with relevant details and sources.
    """
)

QUERY_REWRITER_PROMPT = PromptTemplate.from_template(
    """
    You are a query rewriter that improves search queries based on feedback about previous search results.
    
    Original Question: {question}
    Previous Retrieved Documents: {retrieved_docs}
    Feedback: {feedback}
    
    Based on the feedback about what was missing or insufficient in the previous search,
    rewrite the query to be more specific, targeted, or comprehensive.
    
    The rewritten query should address the gaps identified in the feedback.
    """
) 