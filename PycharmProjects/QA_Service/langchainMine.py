import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType

from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.indexes import Vec
from langchain.evaluation.qa import QAEvalChain

OPENAI_API_KEY = "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" #@param {type:"string"}
HUGGINGFACEHUB_API_TOKEN = "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" #@param {type:"string"}
SERPAPI_API_KEY = "2b01286aa7153594404394e777ac125688f4a1c991bc671c61f3b9fe7020eec1" #@param {type:"string"}

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
os.environ["SERPAPI_API_KEY"] = SERPAPI_API_KEY

chat = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.9)
chat.openai_api_key = OPENAI_API_KEY

# prompt와 chain
prompt = PromptTemplate(
    input_variables=["input"],
    template="{input} 에 대해서 간단하게 설명해줘",
)
prompt.format(input="포맷")
chain = LLMChain(llm=chat, prompt=prompt)
chain.run(input="ai 분야transformer")

# ChatPromptTemplate 와 chain(prompt 디테일 버전)
sys_temp = "You are a helpful assisstant that tranlates {input_language} to {output_language}."
sys_msg_prompt = SystemMessagePromptTemplate.from_template(sys_temp)
human_temp = "{text}"
human_msg_prompt = HumanMessagePromptTemplate.from_template(human_temp)

chat_prompt = ChatPromptTemplate.from_messages([sys_msg_prompt, human_msg_prompt])
deeperChain = LLMChain(llm=chat, prompt=chat_prompt)
deeperChain.run(input_language="English", output_language="Korean", text="I like sushi.")

# agent,tool ??=>agent에 prompt 넣는 법은??
tools = load_tools(tool_names=["wikipedia", "llm-math"], llm=chat)
agent = initialize_agent(tools=tools, llm=chat, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
agent.run("미국 인구 수 더하기 일본 인구수는?")
agent.run("이전의 답에 10을 더하면?") #memory가 없다

# memory




# document 저장하고 임베딩 만들어서 vector DB 만들어서 사용하는 과정

chat = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.9)
chat.openai_api_key = OPENAI_API_KEY

loader = WebBaseLoader(web_path='https://ko.wikipedia.org/wiki/NewJeans')
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings()
index = VectorstoreIndexCreator(vectorstore_cls=FAISS,
                                embedding=embeddings,
                                ).from_loaders([loader])
index.vectorstore.save_local("faiss-nj")
index.query("뉴진스 멤버는?", llm=chat, verbose=True)

fdb = FAISS.load_local("faiss-nj", embeddings)
index2 = VectorStoreIndexWrapper(vectorstore=fdb)

question = "뉴진스 멤버 갈쳐줘"
pred = index2.query(question, llm=chat, verbose=True)



# ------------------------
retriever = index.vectorstore.as_retriever(search_kwargs=dict(k=1))
memory = Vector~~~
examples = [
    {
        "question" : question,
        "answer" : pred
    }
]

# evaluation 하기 위해서는 llmchain 필요(vectorstore -> retriever 생성 -> llm
"""
예시
faiss 통해 vectorstore 생성
retriever = vectorstore.as_retriever(search_kwargs=dict(k=1))
memory = VectorStoreRetrieverMemory(retriever=retriever)
LLMChain(llm=llm, prompt=prompt, verbose=True, memory=memory)
"""
eval_chain = QAEvalChain.from_llm(chat)

