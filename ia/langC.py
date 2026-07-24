import requests
import time
from base64 import b64encode
from typing import Optional
from pydantic import BaseModel, Field
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
#from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import ToolCallLimitMiddleware, ModelCallLimitMiddleware, ModelRequest, ModelResponse, dynamic_prompt, wrap_model_call, AgentMiddleware, AgentState, SummarizationMiddleware
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import create_retriever_tool


#Declarando una funcion ejecutable por la ia (solo por la ia), si esta funcion invocase otro agente podria ser orquestacion de agentes, pero para eso esta mejor Langgraph
@tool('get_weather', description='Returns the current weather in a specific city', return_direct=False)
def get_weather(city: str):
    try:
        response = requests.get(f'https://wttr.in/{city}?format=j1')
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return f"Error fetching weather: {e}"
#print(get_weather("Anadyr")) esto no va




#Creando un conector al modelo de ia, en este caso con LMStudio cuya API es igual que la de OpenAI (dependiendo del modelo y proveedor puede cambiar)
model = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio", model="qwen/qwen2.5-coder-14b", temperature=0.7)
#agent = create_agent(model='gpt-4.1-mini', tools=[get_weather], system_prompt="You are a helpful assistant") #Ejemplo creando el agente con OpenAI como proveedor
agent = create_agent(model=model, tools=[get_weather], system_prompt="You are a helpful assistant") #Creando el agente a partir del modelo
#Usar el modelo enviando un array de mensajes, convendria darle pautas para que interprete y responda de maneras concretas (como esta llamando al modelo, convendria usar llamadas asincronas)
response = agent.invoke({'messages': [
    {'role': 'system', 'content': 'Answer only user questions about weather in different parts of the world making use of the available tools for fetching the data, don\'t respond if the user asks for anything else'},
    #{'role': 'user', 'content': 'Fibonacci in C'} #con esto te dira que no puede
    {'role': 'user', 'content': 'Whats the current temperature in Hanoi?'}
]})
print(response)
print(response['messages'][-1].content)




@dataclass
class Context: #Clase que definira el schema del contexto para el agente
    user_id: str
@dataclass 
class ResponseFormat(BaseModel): #Clase que definira el schema de la respuesta del agente
    chat_response: str = Field(description="The conversational text response to the user.")
    temperature_celsius: Optional[float] = Field(default=None, description="Temperature in Celsius, ONLY if a weather tool was used.")
    humidity: Optional[float] = Field(default=None, description="Humidity percentage, ONLY if a weather tool was used.")
    
@tool('locate_user', description='Find the user\'s city based on their id')
def locate_user(runtime: ToolRuntime[Context]): #Este tool recibe el runtime de la sesion con el agente, como contexto guardamos el id del usuario
    match runtime.context.user_id:
        case '1':
            return 'Monovar'
        case '2':
            return 'Perth'
        case _:
            return 'Unknown'
        

#Establecer modelo de chat, crear un guardador de estado de conversacion (checkpointer) y crear el agente que use los tools y que tenga una estructura de contexto definida (en este caso guarda el id del usuario para ver su ciudad) y un formato de respuesta
tool_limiter = ToolCallLimitMiddleware(run_limit=3, exit_behavior="error") #Middleware para limitar al modelo, hay de mas tipos
modelo_chat = init_chat_model(base_url="http://localhost:1234/v1", api_key="lm-studio", model="qwen/qwen2.5-coder-14b", model_provider='openai', temperature=0.2, max_tokens=8192)
checkpointer = InMemorySaver()
agent = create_agent(model=modelo_chat, tools=[get_weather, locate_user], system_prompt="You are a helpful assistant that helps with hiking, you have access to tools but you should use them only if the task really needs it, if unsure, dont call it and inform the user", context_schema=Context, response_format=ResponseFormat, checkpointer=checkpointer, middleware=[tool_limiter])
config = {'configurable': {'thread_id': '1'}}
response = agent.invoke({'messages': [
    {'role': "user", 'content': "What is the temperature in my city?"}
]}, config=config, context=Context(user_id='1')) #La configuracion tiene el numero de hilo, el contexto es propio para el usuario que interactua, asi se puede tener un agente multiusuario
print(response) #Respuesta segun el schema ya definido
print(response['structured_response'].temperature_celsius)
response = agent.invoke({'messages': [{'role': "user", 'content': "Is it good for going out?"}]}, config=config, context=Context(user_id='1')) #Seguir la conversacion con el contexto anterior
print(response['structured_response'])



#Otra manera de llamar al modelo de forma rapida
modelo_chat = init_chat_model(base_url="http://localhost:1234/v1", api_key="lm-studio", model="qwen/qwen2.5-coder-14b", model_provider='openai', temperature=0.2)
response = modelo_chat.invoke('What is redundancy?')
print(response.content)
#Con un historial de mensajes previo
conversation = [SystemMessage("You are a helpful asssistant"), HumanMessage("What is Langchain?"), AIMessage("Langchain is a library for developing AI agents"), HumanMessage("How do I use it?")]
response = modelo_chat.invoke(conversation)
print(response.content)
#Stream para ir recibiendo chunks del resultado
for chunk in modelo_chat.stream("Count to 50"):
    print(chunk.text, end='', flush=True)



imagen = b64encode(open('imagen.png', 'rb').read()).decode()
modelo_chat = init_chat_model(base_url="http://localhost:1234/v1", api_key="lm-studio", model="qwen/qwen2.5-coder-14b", model_provider='openai', temperature=0.2)
message = HumanMessage(content=[
        {"type": "text", "text": "Describe the contents of this image"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{imagen}"}}
    ])
response = modelo_chat.invoke([message])
print(response.content)


#Modelo de embeddings para RAG, los textos se sacarian de archivos por ejemplo con la debida tecnica de chunking y overlapping
embeddings = OpenAIEmbeddings(model="text-embedding-nomic-embed-text-v1.5-embedding",base_url="http://localhost:1234/v1",api_key="lm-studio",check_embedding_ctx_length=False) #Se esta usando un modelo de embeddings bastante rapido pero no tan eficaz, porque podria relacionar palabras como rest (descansar) y REST (api), en LMStudio existen modelos mas capaces pero tambien mas costosos y lentos
texts = ["For maintaining healthy you must rest at night, otherwise you will be sleeping all day","A REST API with good structure should be able to be up all night without the server sleeping so there is availability","LangChain provides standard interfaces for connecting LLMs with external data sources.","LM Studio allows developers to run open-weight language models locally on consumer hardware.","Python is the dominant programming language for data science and AI engineering workflows.","DDNet is a relaxing game to chill with your friends, beating maps making use of teamwork","The most important skill to beat FakeGame 2 is patience"]
vector_store = FAISS.from_texts(texts, embedding=embeddings) #Base de datos vectorial, tambien se puede usar ChromaDB
print(vector_store.similarity_search("Im very tired, what do you recommend?", k=4)) #Los ejemplos podrian ser mejores y mas grandes (chunking de un documento), pero serviria para analizar un prompt y incrustar solo los chunks mas semanticamente relevantes para no saturar el contexto
retriever = vector_store.as_retriever(search_kwargs={'k': 3}) #Retriever que se usa despues como tool para un agente
retriever_tool = create_retriever_tool(retriever, name='kb_search', description='Search in the document database for information') #Tool normal para un agente
model = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio", model="qwen/qwen2.5-coder-14b", temperature=0.7)
agent = create_agent(model=model, tools=[retriever_tool], system_prompt="You are a helpful assistant that can search in a knowledge database using it as a main source of truth, maybe you have to use it multiple times before answering")
print(agent.invoke({'messages': [{"role": "user", "content": "What is the most important skill to beat FakeGame 2? Patience or fast thinking? Argument it"}]})["messages"][-1].content)

vector_store = Chroma(collection_name="texts", embedding_function=embeddings, persist_directory="./chroma_db_cache") #Ahora con ChromaDB
vector_store.add_texts(texts)
print(vector_store.similarity_search("Im very tired, what do you recommend?", k=3))
print(vector_store.max_marginal_relevance_search("Im very tired, what do you recommend?", k=2, fetch_k=4)) #Hay varios algoritmos




@dataclass
class ContextoDos:
    user_role: str
@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str: #Definiendo un middleware, alterara su comportamiento dependiendo del nivel del usuario (almacenado en contexto)
    user_role = request.runtime.context.user_role
    base_prompt = "You are a helpful assistant"
    match user_role:
        case 'expert':
            return f'{base_prompt}, provide advanced technical responses'
        case 'beginner':
            return f'{base_prompt}, provide basic responses with examples'
        case _:
            return base_prompt
model = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio", model="qwen/qwen2.5-coder-14b", temperature=0.5)
@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse: #Este otro middleware decidiria usar un modelo o otro
    count = len(request.state['messages']) #Se refiere a la cantidad de mensajes, no a la longitud de estos
    if count > 3:
        modelo_usar = model
    else:
        modelo_usar = model #Aqui cada uno apuntaria a modelos distintos
    request.model = modelo_usar
    return handler(request)

agent = create_agent(model=model, middleware=[user_role_prompt, dynamic_model_selection], context_schema=ContextoDos)
print(agent.invoke({'messages': [SystemMessage("Proceed now to answer the user query"), HumanMessage('Explain FTP')]}, context=ContextoDos(user_role='expert'))) #El system prompt varia segun el rol del contexto gracias al middleware

class HooksEjemplo(AgentMiddleware):
    def __init__(self):
        super().__init__()
        self.start_time = 0.0
    def before_agent(self, state: AgentState, runtime): #Hooks que se ejecutan en distintas partes del flujo de interaccion con un modelo, consultar documentacion de Langchain
        self.start_time = time.time()
        print("before_agent")
    def before_model(self, state: AgentState, runtime):
        print("before_model")
    def after_model(self, state: AgentState, runtime):
        print("after_model")
    def after_agent(self, state: AgentState, runtime):
        print("after_agent")
        print(time.time() - self.start_time)
    
model = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio", model="qwen/qwen2.5-coder-14b", temperature=0.4)
agent = create_agent(model=model, middleware=[HooksEjemplo()])
print(agent.invoke({'messages': [SystemMessage("You are a helpful assistant"), HumanMessage("Fibonacci in C")]}))
agent = create_agent(model=model, middleware=[SummarizationMiddleware(model=model, max_tokens_before_summary=4000, messages_to_keep=20, summary_prompt="Summarize the most important parts of the conversation")]) #Otro middleware que en este caso podria servir para compactar contextos, hay mas middlewares ya hechos

