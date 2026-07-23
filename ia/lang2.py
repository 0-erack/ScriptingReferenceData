import requests
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import ToolCallLimitMiddleware, ModelCallLimitMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from langchain_community.vectorstores import FAISS
from base64 import b64encode


#Declarando una funcion ejecutable por la ia (solo por la ia)
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
model = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio", model="local-model", temperature=0.7)
#agent = create_agent(model='gpt-4.1-mini', tools=[get_weather], system_prompt="You are a helpful assistant") #Ejemplo creando el agente con OpenAI como proveedor
agent = create_agent(model=model, tools=[get_weather], system_prompt="You are a helpful assistant") #Creando el agente a partir del modelo
#Usar el modelo enviando un array de mensajes, convendria darle pautas para que interprete y responda de maneras concretas
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
tool_limiter = ToolCallLimitMiddleware(run_limit=3, exit_behavior="error")
modelo_chat = init_chat_model(base_url="http://localhost:1234/v1", api_key="lm-studio", model="local-model", model_provider='openai', temperature=0.2, max_tokens=8192)
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
modelo_chat = init_chat_model(base_url="http://localhost:1234/v1", api_key="lm-studio", model="local-model", model_provider='openai', temperature=0.2)
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
modelo_chat = init_chat_model(base_url="http://localhost:1234/v1", api_key="lm-studio", model="local-model", model_provider='openai', temperature=0.2)
message = HumanMessage(content=[
        {"type": "text", "text": "Describe the contents of this image"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{imagen}"}}
    ])
response = modelo_chat.invoke([message])
print(response.content)



