from datetime import datetime
import os, sys, platform
import langchain, langchain_core
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import ast
import operator as op
from typing import Union
#from langchain.agents import AgentExecutor, create_tool_calling_agent

#TODO
