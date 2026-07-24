import uuid
from typing import TypedDict, Annotated, Literal
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages


#Se necesita un modelo, proveido por Langchain
llm = init_chat_model(base_url="http://localhost:1234/v1", api_key="lm-studio", model="google/gemma-4-12b-qat", model_provider='openai', temperature=0.2)

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


#Clasificar la intencion del mensaje inicial y dividir el grafo en varios nodos
class IntentClassifier(BaseModel): #El output que debe devolver el modelo clasificador
    message_intent: Literal['chat', 'knowledge', 'code'] = Field(..., description="Classify wether the user wants to just chat, ask for knowledge or change code in the project")
class State(TypedDict): #Esto es como MessagesState, estado custom para pasar entre nodos
    messages = Annotated[list, add_messages]
    message_intent = str | None
def classify_intent(state: State): #Nodo clasificador, hace que el LLM determine la intencion del prompt del usuario devolviendolo en el campo personalizado del State, el modelo devuelve en formato IntentClassifier
    structured_llm = llm.with_structured_output(IntentClassifier)
    result = structured_llm.invoke([{'role': 'system', 'content': 'Determine wether the user wants to "chat", retrieve "knowledge" or change "code"'}, {'role': 'user', 'content': state.messages[-1].content}])
    return {'message_intent': result.message_intent}
#Nodos siguientes de la decision segun intencion
def prompt_llm_chat(state: State):
    messages = [{'role': 'system', 'content': 'You are a talkative chatbot'}] + state['messages']
    response = llm.invoke(messages)
    return {'messages': [{'role': 'assistant', 'content': response.content}]}
def prompt_llm_chat(state: State):
    messages = [{'role': 'system', 'content': 'You are a talkative chatbot'}] + state['messages']
    response = llm.invoke(messages)
    return {'messages': [{'role': 'assistant', 'content': response.content}]}
def prompt_llm_chat(state: State):
    messages = [{'role': 'system', 'content': 'You are a talkative chatbot'}] + state['messages']
    response = llm.invoke(messages)
    return {'messages': [{'role': 'assistant', 'content': response.content}]}

