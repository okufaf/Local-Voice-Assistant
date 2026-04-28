import os

from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful and friendly AI assistant. 
    You are polite, respectful, and aim to provide concise responses of less than 50 words."""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

llm = ChatOpenAI(
    model="minimax/minimax-m2.5:free",
    temperature=0.6,
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    timeout=15
)

chain = prompt | llm

chat_sessions = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """
    Get or create chat history for a session.

    Args:
        session_id (str): Unique session identifier.

    Returns:
        InMemoryChatMessageHistory: The chat history object.
    """
    if session_id not in chat_sessions:
        chat_sessions[session_id] = InMemoryChatMessageHistory()
    return chat_sessions[session_id]


chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)


def get_llm_response(text: str) -> str:
    """
    Generates a response to the given text using the language model.

    Args:
        text (str): The input text to be processed.

    Returns:
        str: The generated response.
    """
    session_id = "voice_assistant_session"
    response = chain_with_history.invoke(
        {"input": text},
        config={"session_id": session_id}
    )
    return (response.content or "").strip()
