import uuid
import os
import subprocess
from typing import TypedDict, Annotated, Literal
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


#Se necesita un modelo, proveido por Langchain
llm = init_chat_model(base_url="http://localhost:1234/v1", api_key="lm-studio", model="google/gemma-4-12b-qat", model_provider='openai', temperature=0.2)
"""
#Nodo del grafo, recibe el estado actual de mensajes (conteniendo los mensajes) y agrega el nuevo que da el modelo
def prompt_llm(state: MessagesState):
    response = llm.invoke(state['messages'])
    return {'messages': [response]}

graph_builder = StateGraph(MessagesState) #Definiendo el grafo
graph_builder.add_node("chatbot", prompt_llm) #Agnadiendo el nodo (simple que solo sigue una conversacion) al grafo
graph_builder.add_edge(START, "chatbot") #Uniendo los nodos
graph_builder.add_edge("chatbot", END)
checkpointer = InMemorySaver() #Guarda el contexto en memoria ram
graph = graph_builder.compile(checkpointer=checkpointer) #El grafo se tiene que compilar antes
config = {'configurable': {'thread_id': uuid.uuid4()}} #Hacer una sesion para ejecutar
while False: #Interactuando iterativamente
    print(graph.invoke({'messages': [{'role': 'user', 'content': input("X: ")}]}, config=config)['messages'][-1].content)


"""

#Clasificar la intencion del mensaje inicial y dividir el grafo en varios nodos
class IntentClassifier(BaseModel): #El output que debe devolver el modelo clasificador
    message_intent: Literal['chat', 'knowledge', 'code'] = Field(..., description="Classify wether the user wants to just chat, ask for knowledge or change code in the project")
class State(TypedDict): #Esto es como MessagesState, estado custom para pasar entre nodos
    messages: Annotated[list, add_messages]
    message_intent: str | None
def classify_intent(state: State): #Nodo clasificador, hace que el LLM determine la intencion del prompt del usuario devolviendolo en el campo personalizado del State, el modelo devuelve en formato IntentClassifier
    structured_llm = llm.with_structured_output(IntentClassifier)
    result = structured_llm.invoke([{'role': 'system', 'content': 'Determine wether the user wants to "chat", retrieve "knowledge" (normally related to questions or asking information), or change "code" (for example, editting or reading local files), return only one of these words'}, {'role': 'user', 'content': state['messages'][-1].content}])
    return {'message_intent': result.message_intent}

#Nodos siguientes de la decision segun intencion (este es si quiere hablar)
def prompt_llm_chat(state: State):
    messages = [{'role': 'system', 'content': 'You are a talkative chatbot'}] + state['messages']
    response = llm.invoke(messages)
    return {'messages': [{'role': 'assistant', 'content': response.content}]}

#Nodo por si quiere buscar informacion (RAG)
KNOWLEDGE = ["For maintaining healthy you must rest at night, otherwise you will be sleeping all day","A REST API with good structure should be able to be up all night without the server sleeping so there is availability","LangChain provides standard interfaces for connecting LLMs with external data sources.","LM Studio allows developers to run open-weight language models locally on consumer hardware.","Python is the dominant programming language for data science and AI engineering workflows.","DDNet is a relaxing game to chill with your friends, beating maps making use of teamwork","Seikravseya is an obscure thriller novel book","The most important skill to beat FakeGame 2 is patience"]
embeddings = OpenAIEmbeddings(model="text-embedding-nomic-embed-text-v1.5-embedding",base_url="http://localhost:1234/v1",api_key="lm-studio",check_embedding_ctx_length=False)
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents([Document(page_content=text) for text in KNOWLEDGE])
def prompt_llm_rag(state: State):
    query = state['messages'][-1].content
    documents = vector_store.similarity_search(query, k=3) #Se esta buscando documentos ya chunkeados y hechos los embeddings, podria ser necesario hacer esto en el nodo en caso de que el usuario ponga sus archivos
    context = '\n'.join(f'- {doc.page_content}' for doc in documents)
    #messages = [{'role': 'system', 'content': 'Only say "I am the RAG agent placeholder"'}] + state['messages']
    print(context)
    messages = [{'role': 'system', 'content': f'You are a RAG agent, answer using only the text below, if the answer is not in it say you dont know.\n\nContext:\n{context}'}] + state['messages']
    response = llm.invoke(messages)
    return {'messages': [{'role': 'assistant', 'content': response.content}]}

#Nodo por si quiere editar codigo
def prompt_llm_code(state: State):
    user_prompt = state['messages'][-1].content
    workspace = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'workspace')
    result = subprocess.run(['echo', user_prompt], cwd=workspace, capture_output=True, text=True) # TODOCAMBIAR PONER TOOL IO
    output = result.stdout.strip() or result.stderr.strip()
    #messages = [{'role': 'system', 'content': 'Only say "I am the coding agent placeholder"'}] + state['messages']
    #response = llm.invoke(messages)
    #return {'messages': [{'role': 'assistant', 'content': response.content}]}
    return {'messages': [{'role': 'assistant', 'content': output}]}

#Configurar el grafo
graph_builder = StateGraph(State)
graph_builder.add_node("classifier", classify_intent)
graph_builder.add_node("chat_agent", prompt_llm_chat)
graph_builder.add_node("rag_agent", prompt_llm_rag)
graph_builder.add_node("code_agent", prompt_llm_code)
graph_builder.add_edge(START, "classifier")
#Nodo condicional, dependiendo de state.message_intent (establecido por el clasificador) ira a un nodo u otro, usando una lambda
graph_builder.add_conditional_edges("classifier", lambda state: state['message_intent'], {'chat': 'chat_agent', 'knowledge': 'rag_agent', 'code': 'code_agent'})
graph_builder.add_edge("chat_agent", END)
graph_builder.add_edge("rag_agent", END)
graph_builder.add_edge("code_agent", END)
checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)
config = {'configurable': {'thread_id': uuid.uuid4()}}
while True: #El grafo se ejecutara de principio a fin haciendo la decision clasificadora por cada iteracion, pudiendo hacer un camino de nodos distinto al del mensaje anterior
    print(graph.invoke({'messages': [{'role': 'user', 'content': input("X: ")}]}, config=config)['messages'][-1].content)
    
