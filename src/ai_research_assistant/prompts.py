"""Prompt templates."""

from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = (
    "You are an AI research assistant. Answer the user's question using ONLY the context "
    "below, which comes from their PDF document. If the answer is not in the context, say "
    "you could not find it in the document. Be clear and concise, and cite the page numbers "
    "you used like (p. 12).\n\nContext:\n{context}"
)

QA_PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{question}")])
