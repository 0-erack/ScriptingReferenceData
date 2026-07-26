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
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langgraph.types import interrupt, Command


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





#Tools para el subagente de codigo
WORKSPACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'workspace')
os.makedirs(WORKSPACE_DIR, exist_ok=True)
@tool('list_files', description="List all files available in the workspace directory.")
def list_files() -> str:
    files = []
    for root, _, filenames in os.walk(WORKSPACE_DIR):
        for f in filenames:
            files.append(os.path.relpath(os.path.join(root, f), WORKSPACE_DIR))
    return str(files) if files else "Workspace is empty."
@tool('read_file', description="Read the contents of a specific file inside the workspace.")
def read_file(file_path: str) -> str:
    full_path = os.path.join(WORKSPACE_DIR, file_path)
    if not os.path.exists(full_path):
        return f"Error: File '{file_path}' does not exist."
    with open(full_path, 'r', encoding='utf-8') as f:
        return f.read()
@tool('edit_file', description="Create or overwrite a file in the workspace with new content.")
def edit_file(file_path: str, content: str) -> str:
    full_path = os.path.join(WORKSPACE_DIR, file_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, 'w', encoding='utf-8') as f:
        f.write(content)
    return f"Successfully updated file '{file_path}'."
@tool('run_bash_command', description="Execute a bash shell command inside the workspace directory. Returns both standard output (stdout) and standard error (stderr). Use it when the user or the task needs to execute a command")
def run_bash_command(command: str) -> str:
    try:
        result = subprocess.run(command,shell=True,cwd=WORKSPACE_DIR,capture_output=True,text=True,timeout=60)
        output = ""
        if result.stdout:
            output += f"--- STDOUT ---\n{result.stdout}\n"
        if result.stderr:
            output += f"--- STDERR ---\n{result.stderr}\n"    
        if not output.strip():
            return "Command executed successfully with no output."
        return output
    except subprocess.TimeoutExpired:
        return "Error: The command timed out after 60 seconds."
    except Exception as e:
        return f"Error executing command: {str(e)}"

#Clasificar la intencion del mensaje inicial y dividir el grafo en varios nodos
class IntentClassifier(BaseModel): #El output que debe devolver el modelo clasificador
    message_intent: Literal['chat', 'knowledge', 'code'] = Field(..., description="Classify wether the user wants to just chat, ask for knowledge or change code in the project")
class State(TypedDict): #Esto es como MessagesState, estado custom para pasar entre nodos
    messages: Annotated[list, add_messages]
    message_intent: str | None
    next_node: str | None
def classify_intent(state: State): #Nodo clasificador, hace que el LLM determine la intencion del prompt del usuario devolviendolo en el campo personalizado del State, el modelo devuelve en formato IntentClassifier
    structured_llm = llm.with_structured_output(IntentClassifier)
    result = structured_llm.invoke([{'role': 'system', 'content': "You are a precise routing classifier. Your job is to analyze the user\'s message and categorize it into EXACTLY ONE of these three intents:\n\n1. \'chat\': \n   - Use for casual greetings, personal opinions, small talk, or general conversation.\n   - Examples: \"Hello\", \"How are you?\", \"Tell me a joke.\"\n\n2. \'knowledge\': \n   - Use when the user asks factual questions, looks up information, or queries the knowledge base/database about specific facts, games, books, or data.\n   - Examples: \"What is the most important skill for FakeGame 2?\", \"Tell me about Seikravseya.\"\n\n3. \'code\': \n   - Use when the user wants to interact with files, look at code, check directories, modify scripts, or create new files.\n   - Examples: \"Show me the files in the workspace\", \"Create a python script for a calculator\", \"Read main.py.\"\n\nReturn ONLY one of these three exact words: chat, knowledge, or code."}, {'role': 'user', 'content': state['messages'][-1].content}])
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

#Nodo para human in the loop, para aprovar o no operaciones de manipulacion de archivos o ejecucion de comandos en el nodo de codigo
def accept_coding(state: State):
    user_prompt = state['messages'][-1].content
    decision = interrupt(f'About to edit code files or execute a command with request\n\n{user_prompt}\n\nApprove? (y/n, or type a revised request)')
    text = str(decision).strip().lower()
    if text in ['y', 'yes', 'si', 'ok']:
        return {}
    if text in ['n', 'no', 'quit', 'exit']:
        return {'messages': [{'role': 'assistant', 'content': 'Coding request was denied by user.'}], 'next_node': 'denied'} #Alterando el estado, este campo indica si procede a realizar los cambios o va al nodo de denegacion
    return {'messages': [{'role': 'user', 'content': "Original task changed, do this instead: " + str(decision)}]}
#Nodo por si quiere editar codigo
def prompt_llm_code(state: State):
    #Hace un agente con tools de manipulacion de archivos, lo interesante seria que se ejecutase en bucle iterando hasta conseguir el resultado, invocando subagentes y compactando contexto
    code_agent_subgraph = create_react_agent(model=llm, tools=[edit_file, read_file, list_files, run_bash_command], prompt="You are an expert coding assistant with access to local files via tools and execute commands. Use the provided tools to inspect, read, and edit files in the workspace as requested by the user.")
    result = code_agent_subgraph.invoke({"messages": state['messages']})
    new_messages = result['messages'][len(state['messages']):]
    return {'messages': new_messages}

#Configurar el grafo
graph_builder = StateGraph(State)
graph_builder.add_node("classifier", classify_intent)
graph_builder.add_node("chat_agent", prompt_llm_chat)
graph_builder.add_node("rag_agent", prompt_llm_rag)
graph_builder.add_node("code_agent", prompt_llm_code)
graph_builder.add_node("accept_coding", accept_coding)
graph_builder.add_edge(START, "classifier")
#Nodo condicional, dependiendo de state.message_intent (establecido por el clasificador) ira a un nodo u otro, usando una lambda
graph_builder.add_conditional_edges("accept_coding", lambda state: 'end' if state.get('next_node') == 'denied' else 'code_agent') #Condicional dependiendo del resultado en el estado de aprovar o no una operacion
graph_builder.add_conditional_edges("classifier", lambda state: state['message_intent'], {'chat': 'chat_agent', 'knowledge': 'rag_agent', 'code': 'accept_coding'}) #Condicional dependiendo de la intencion del prompt del usuario
graph_builder.add_edge("chat_agent", END)
graph_builder.add_edge("rag_agent", END)
graph_builder.add_edge("code_agent", END)
checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer) #Finalmente compilar el grafo
#graph.get_graph().draw_mermaid_png(output_file_path="graph.png") #Se puede visualizar como imagen
config = {'configurable': {'thread_id': uuid.uuid4()}}
while True: #El grafo se ejecutara de principio a fin haciendo la decision clasificadora por cada iteracion, pudiendo hacer un camino de nodos distinto al del mensaje anterior
    #print(graph.invoke({'messages': [{'role': 'user', 'content': input("X: ")}]}, config=config)['messages'][-1].content)
    user_message = input("X: ")
    result = graph.invoke({'messages': [{'role': 'user', 'content': user_message}]}, config=config)
    while '__interrupt__' in result: #Comprobando el resultado de la interrupcion del human in the loop
        prompt = result['__interrupt__'][0].value
        decision = input(f'{prompt}\n> ')
        result = graph.invoke(Command(resume=decision), config=config)
    print(result['messages'][-1].content)
    

