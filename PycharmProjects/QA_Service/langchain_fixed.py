#@title 0. API 키 설정
import os


OPENAI_API_KEY = "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" #@param {type:"string"}

HUGGINGFACEHUB_API_TOKEN = "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" #@param {type:"string"}

SERPAPI_API_KEY = "2b01286aa7153594404394e777ac125688f4a1c991bc671c61f3b9fe7020eec1" #@param {type:"string"}

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
os.environ["SERPAPI_API_KEY"] = SERPAPI_API_KEY



#@title 2. ChatOpenAI LLM (gpt-3.5-turbo)
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

chat = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.9)
chat.openai_api_key = OPENAI_API_KEY
sys = SystemMessage(content="당신은 음악 추천을 해주는 전문 AI입니다.")
msg = HumanMessage(content='1980년대 메탈 음악 5곡 추천해줘.')

aimsg = chat([sys, msg])
aimsg.content

#@title 3. Prompt Template & chain

from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["상품"],
    template="{상품} 만드는 회사 이름 추천해줘. 기억에 남는 한글 이름으로",
)

prompt.format(상품="AI 여행 추천 서비스")

from langchain.chains import LLMChain
chain = LLMChain(llm=chat, prompt=prompt)

# chain.run("AI 여행 추천 서비스")
chain.run(상품="AI 여행 추천 서비스")


#@title 4. ChatPromptTemplate & chain

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

chat = ChatOpenAI(temperature=0)

template="You are a helpful assisstant that tranlates {input_language} to {output_language}."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template="{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])


chatchain = LLMChain(llm=chat, prompt=chat_prompt)
chatchain.run(input_language="English", output_language="Korean", text="I love programming.")

#@title 5. Agents and Tools

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType

# tools = load_tools(["serpapi", "llm-math"], llm=chat)
tools = load_tools(["wikipedia", "llm-math"], llm=chat)

agent = initialize_agent(tools, llm=chat, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
agent.run("페이스북 창업자는 누구인지? 그의 현재(2023년) 나이를 제곱하면?")
agent.tools

print(agent.tools[0].description)
print(agent.tools[1].description)


#@title 6. Memory
from langchain import ConversationChain

conversation = ConversationChain(llm=chat, verbose=True)
conversation.predict(input="인공지능에서 Transformer가 뭐야?")
conversation.predict(input="RNN하고 차이 설명해줘.")
conversation.predict(input="attention은 뭐야?")
conversation.predict(input="context vector 쓰는 건 가?")
conversation.predict(input="attention 과 context vector의 연관성은?")
conversation.predict(input="뉴진스 알아?")
conversation.predict(input="kpop 걸그룹 뉴진스의 데뷔곡은?")
conversation.memory

# 여기서 부터는 문서를 저장하고 임베딩 만들어서 VectorDB에 저장하고 사용하는 과정

#@title 7. Document Loaders
from langchain.document_loaders import WebBaseLoader

loader = WebBaseLoader(web_path="https://ko.wikipedia.org/wiki/NewJeans")
documents = loader.load()

from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# 4096 token = 3000 English word

print(docs[1].page_content)
docs[2].page_content

#@title 8. Summarization
# from langchain.chains.summarize import load_summarize_chain
# chain = load_summarize_chain(chat, chain_type="map_reduce", verbose=True)
# chain.run(docs[:3])

#@title 9. Embeddings and VectorStore
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import OpenAIEmbeddings

# embeddings = OpenAIEmbeddings()
embeddings = HuggingFaceEmbeddings()

from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS

# from langchain.text_splitter import RecursiveCharacterTextSplitter
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

index = VectorstoreIndexCreator(
    vectorstore_cls=FAISS,
    embedding=embeddings,
    # text_splitter=text_splitter,
    ).from_loaders([loader])

# 파일로 저장
index.vectorstore.save_local("faiss-nj")

# 질문해보기
import math
import time

# 3.7초
index.query("뉴진스의 데뷔곡은?", llm=chat, verbose=True)


# 15초
start = time.time()
index.query("뉴진스 설명해줘", llm=chat, verbose=True)
end = time.time()
print(end-start)

# 15초 > 23초 (시간 유의미하지 않음)
start = time.time()
index.query("블랙핑크 설명해줘", llm=chat, verbose=True)
end = time.time()
print(end-start)

#@title FAISS 벡터DB 디스크에서 불러오기
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

fdb = FAISS.load_local("faiss-nj", embeddings)
index2 = VectorStoreIndexWrapper(vectorstore=fdb)
index2.query("뉴진스의 데뷔 멤버는?", llm=chat, verbose=True)