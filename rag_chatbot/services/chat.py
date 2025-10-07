from itertools import tee
import opik
from ..services.vectorstore import get_vectorstore, get_vectorstore_by_type
from langchain_openai import ChatOpenAI
from ..config import Config
import logging
import os
from langchain.schema import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


def format_chat_history(messages):
    """
    Convert chat history messages into a formatted string.
    
    Args:
        messages: List of HumanMessage and AIMessage objects.
        
    Returns:
        Formatted chat history as a string.
    """
    formatted_history = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            formatted_history.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted_history.append(f"AI: {msg.content}")
    return "\n".join(formatted_history)


def extract_metadata(doc):
    """
    Extract and normalize metadata from a document.
    
    Args:
        doc: The document containing metadata.
        
    Returns:
        A dictionary with normalized metadata.
    """
    metadata = doc.metadata
    
    # Extract source with fallbacks
    source = metadata.get("source", metadata.get("filename", "Unknown"))
    
    # Handle page number with proper type conversion
    page_number = None
    if "page" in metadata:
        try:
            # Convert to integer and ensure it's at least 1
            page_number = metadata["page"] + 1
        except (ValueError, TypeError):
            # Default to None if conversion fails
            logger.debug(f"Failed to convert page number: {metadata['page_number']}")
            page_number = None
    
    # Build source info dictionary
    source_info = {
        "source": source,
        "page_number": page_number if page_number is not None else 1  # Default to page 1 if missing
    }
    
    # Add source URL for markdown files
    if "source_url" in metadata:
        source_info["source_url"] = metadata["source_url"]
    
    # Add additional useful metadata
    if "original_filename" in metadata:
        source_info["original_filename"] = metadata["original_filename"]
    
            
    return source_info


def get_answer(
    question: str, 
    responseType: str, 
    chat_history, 
    model_name: str, 
    provider: str = "openai",
    vectorstore_type: str = "knowledge_base"
):
    """
    Get an answer to a query using the RAG system
    
    Args:
        question: The user's query
        responseType: The type of response, either "concise" or "expressive"
        chat_history: The chat history
        model_name: The name of the model to use
        provider: The LLM provider to use (default is "openai")
        vectorstore_type: The type of vectorstore to use, either "knowledge_base", "samsung", or "sonu" (default is "knowledge_base")

    Returns:
        A dictionary containing the answer and sources
        
    Raises:
        ValueError: If the vector store is not initialized or the LLM provider is unsupported
    """
    # Get the vector store based on type
    vectorstore = get_vectorstore_by_type(vectorstore_type)
    logger.info(f"Vectorstore type:::::chat: {vectorstore_type}")
    if vectorstore is None:
        logger.error("Vector store is not initialized")
        raise ValueError("Vector store is not initialized. Please check the logs for more information.")
    
    concise = responseType == "concise"

    try:
        openai_api_key = Config.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        google_api_key = os.getenv("GOOGLE_API_KEY")

        # Create retriever
        logger.info("Creating retriever")
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.3, "k": 3}
        )        
        
        # Initialize LLM based on provider
        llm_mapping = {
            "openai": lambda: ChatOpenAI(model_name=model_name, api_key=openai_api_key),
            "anthropic": lambda: ChatAnthropic(model_name=model_name, api_key=anthropic_api_key),
            "deepseek": lambda: ChatDeepSeek(model_name=model_name, api_key=deepseek_api_key),
            "google": lambda: ChatGoogleGenerativeAI(model=model_name, api_key=google_api_key)
        }
        
        if provider not in llm_mapping:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        
        logger.info(f"Initializing {provider.capitalize()} LLM with model: {model_name}")
        llm = llm_mapping[provider]()
        
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )
        
        # Create QA chain with memory
        logger.info(f"Creating QA chain with memory for {vectorstore_type} vectorstore") 
        # Define system prompts based on vectorstore type
        if vectorstore_type.lower() == "sonu":
            qa_system_prompt_expressive = """Lets roleplay, You are an assistant on `soundhealth.life` website. SoundHealth is a medical technology company harnessing the power of artificial intelligence and medical science to improve respiratory health. The SoundHealth team consists of experienced medical professionals, data scientists and engineers who are passionate about improving healthcare.
            Answer questions based on the context given from Knowledge Base,
            there is only one product named `sonu` in different variant.
            Always use markdown formatting. Always follow these formatting rules:
            - Use **bold** text for emphasis
            - Use *italic* for subtle emphasis
            - Use ~~strikethrough~~ for obsolete or incorrect items
            - Use <u>underline</u> for underlined text (since Markdown doesn't support it natively)
            - Use `#`, `##`, `###` etc. for headings
            - Use bullet points (`-`) for lists
            - always Use `<br>` for new lines
            - Use **blank lines** (double newline) between paragraphs
            - Use paragraphs where needed

            Do not answer questions that are not related to the Sound Health or the context given.
            If you do not have the information, ask them to contact support on this URL: https://soundhealth.life/pages/contact.

            ## Always Remember:
            Ensure responses are helpful, expressive, and aligned with the user's intent. Keep tone professtional and maintain a natural dialogue flow. Do not use any emojis.
            <context>
            {context}
            </context>
            """
        else:  # Samsung prompts
            qa_system_prompt_expressive = """
            You are Customer Support Assistant that can answer questions about the context given,         
            You are an AI assistant that answers questions based exclusively on the context fetched from a retriever. You must not use any external knowledge, make assumptions, or attempt to search for additional information. Provide responses only when the context explicitly contains the necessary information. If it does not, respond with not having that information.
            You are having manuals of all the Samsung appliances, so ask the user to provide the model number of the appliance they need help with. clarify this with the user before answering the question.
            You represent the company Samsung, so use the name Samsung in your responses. you are having manuals of all the Samsung appliances, so ask the user to provide the model number of the appliance they need help with. clarify this with the user before answering the question.

            ## Personality Traits:
            - Friendly and approachable
            - Empathetic and emotionally intelligent
            - Knowledgeable but not pretentious
            - Patient and helpful
            
            ## Always use markdown formatting. Always follow these formatting rules:

            - Use **bold** text for emphasis
            - Use *italic* for subtle emphasis
            - Use ~~strikethrough~~ for obsolete or incorrect items
            - Use <u>underline</u> for underlined text (since Markdown doesn't support it natively)
            - Use `#`, `##`, `###` etc. for headings
            - Use bullet points (`-`) for lists
            - always Use `<br>` for new lines
            - Use **blank lines** (double newline) between paragraphs
            - Use paragraphs where needed
            
            ## Always Remember:
            Ensure responses are helpful, expressive, and aligned with the user's intent. Keep responses conversational, and maintain a natural dialogue flow. Be friendly, approachable, and engaging. Adapt personality and responses based on the user's style and context. Do not break persona. Create a seamless, engaging conversation that feels human.
            <context>
            {context}
            </context>
            """

        if vectorstore_type.lower() == "sonu":
            qa_system_prompt_concise = """Lets roleplay, You are an assistant on `soundhealth.life` website. SoundHealth is a medical technology company harnessing the power of artificial intelligence and medical science to improve respiratory health. The SoundHealth team consists of experienced medical professionals, data scientists and engineers who are passionate about improving healthcare.
            Answer questions based on the context given from Knowledge Base,
            
            Always use markdown formatting. Always follow these formatting rules:
            - Use **bold** text for emphasis
            - Use *italic* for subtle emphasis
            - Use ~~strikethrough~~ for obsolete or incorrect items
            - Use <u>underline</u> for underlined text (since Markdown doesn't support it natively)
            - Use `#`, `##`, `###` etc. for headings
            - Use bullet points (`-`) for lists
            - always Use `<br>` for new lines
            - Use **blank lines** (double newline) between paragraphs
            - Use paragraphs where needed

            Do not answer questions that are not related to the Sound Health or the context given.
            If you do not have the information, ask them to contact support on this URL: https://soundhealth.life/pages/contact.

            ## Always Remember:
            Ensure responses are helpful, expressive, and aligned with the user's intent. Keep tone professtional and maintain a natural dialogue flow. Do not use any emojis.
            <context>
            {context}
            </context>
            """
        else:  # Samsung prompts
            qa_system_prompt_concise ="""You are Customer Support Assistant that can answer questions about the context given,         
            You are an Customer Support assistant that answers questions based exclusively on the context fetched from a retriever. You must not use any external knowledge, make assumptions, or attempt to search for additional information. Provide responses only when the context explicitly contains the necessary information. If it does not, respond with not having that information.
            You are having manuals of all the Samsung appliances, so ask the user to provide the model number of the appliance they need help with. clarify this with the user before answering the question.
            You represent the company Samsung, so use the name Samsung in your responses.you are having manuals of all the Samsung appliances, so ask the user to provide the model number of the appliance they need help with. clarify this with the user before answering the question.
            if you are asked to help with appliances before denying make sure to ask the user to provide the model number of the appliance they need help with.
            ## Personality Traits:
            - Friendly and approachable
            - Empathetic and emotionally intelligent
            - Knowledgeable but not pretentious
            - Patient and helpful
            
            ## Always Remember:
            Ensure responses are helpful, concise, and aligned with the user's intent. Keep responses conversational, concise, and maintain a natural dialogue flow. Be friendly, approachable, and engaging. Adapt personality and responses based on the user's style and context. Do not break persona. Create a seamless, engaging conversation that feels human.generate answers use atmost 50 words.
            
            <context>
            {context}
            </context>
            """

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt_concise if concise else qa_system_prompt_expressive),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        
        # Retrieve documents
        docs = history_aware_retriever.invoke({"input": question, "chat_history": chat_history})
        
        response = question_answer_chain.invoke({
                "input": question,
                "chat_history": chat_history,
                "context": docs
        })
        answer = response

        # Extract metadata from retrieved documents
        detailed_sources = [extract_metadata(doc) for doc in docs]
        
        # Log the retrieved sources with page numbers for debugging
        logger.info(f"Sources retrieved: {detailed_sources}")
        logger.info(f"Answer: {answer}")
        
        return {
            "answer": answer,
            "query": question,
            "chat_history": chat_history,
            "sources": detailed_sources
        }
    except Exception as e:
        logger.error(f"Error getting answer: {str(e)}", exc_info=True)
        raise