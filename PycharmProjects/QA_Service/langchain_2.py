#@title 0. API 키 설정
import os
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.prompts import PromptTemplate #미리 설정해두는 문구, 구조, tf로 치면 placeholder같은 느낌
from langchain.chains import LLMChain

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplte,
    HumanMessagePromptTemplate,
)

OPENAI_API_KEY = "sk-" #@param {type:"string"}
HUGGINGFACEHUB_API_TOKEN = "" #@param {type:"string"}
SERPAPI_API_KEY = "" #@param {type:"string"}

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
os.environ["SERPAPI_API_KEY"] = SERPAPI_API_KEY




#@title 2. ChatOpenAI LLM (gpt-3.5-turbo)
chat = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.9) #chat은 gpt-turbo로 설정하는 부분
chat.openai_api_key = OPENAI_API_KEY

sys = SystemMessage(content="You should explain me about wine.")
msg = HumanMessage(content="Recommend Burgundy wine")

AImsg = chat([sys, msg])
AImsg.content

#@title 3. Prompt Template & chain
prompt = PromptTemplate(
    input_variables=["상품"],
    template="{상품} 만드는 회사 이름 추천해줘. 고급스러운 느낌으로",
)

# prompt.format(상품="고급 막걸리")
chain = LLMChain(llm=chat, prompt=prompt)

# chain.run("AI 여행 추천 서비스")
chain.run(상품="막걸리")


#@title 4. ChatPromptTemplate & chain


template="You are a helpful assisstant that tranlates {input_language} to {output_language}."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_message_prompt = HumanMessagePromptTemplate.from_template("{text}")
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

chat = ChatOpenAI(temperature=0)

chatchain = LLMChain(llm=chat, prompt=chat_prompt)
chatchain.run(input_language="English", output_language="Korean", text="I love you.")

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

conversation = ConversationChain(llm=chat, verbose=True) #기본으로 basememory, baseprompt 사용 / prompt 구성은 어떻게?
conversation.predict(input="인공지능에서 Transformer가 뭐야?")
conversation.predict(input="RNN하고 차이 설명해줘.")
conversation.predict(input="attention은 뭐야?")
conversation.predict(input="context vector 쓰는 건 가?")
conversation.predict(input="attention 과 context vector의 연관성은?")
conversation.predict(input="뉴진스 알아?")
conversation.predict(input="아니 kpop 걸그룹 뉴진스")
conversation.predict(input="newJeans의 데뷔곡 알아?")
conversation.predict(input="블랙핑크 데뷔곡 알아?")
conversation.predict(input="22년에 데뷔한 kpop 걸그룹 알아?")
conversation.predict(input="그 말은 22년 데이터를 학습하지 않았다는 뜻이네??")
conversation.predict(input="그럼 몇 년도까지 정보를 제공할 수 있는데?")
conversation.memory

# 여기서 부터는 문서를 저장하고 임베딩 만들어서 VectorDB에 저장하고 사용하는 과정
#@title 7. Document Loaders
from langchain.document_loaders import WebBaseLoader
loader = WebBaseLoader(web_path="https://ko.wikipedia.org/wiki/NewJeans")
documents = loader.load()

from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
len(docs)

# 4096 token = 3000 English word

print(docs[1].page_content)



#@title 8. Summarization
from langchain.chains.summarize import load_summarize_chain
chain = load_summarize_chain(chat, chain_type="map_reduce", verbose=True)
chain.run(docs[:3])

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

index.query("뉴진스의 데뷔곡은?", llm=chat, verbose=True)
index.query("멤버의 나이는?", llm=chat, verbose=True)


#@title FAISS 벡터DB 디스크에서 불러오기
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

fdb = FAISS.load_local("faiss-nj", embeddings)
index2 = VectorStoreIndexWrapper(vectorstore=fdb)
