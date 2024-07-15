import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()
embedding = OpenAIEmbeddings(model='text-embedding-3-large')
index_name = 'cheekotalk'
llm = ChatOpenAI(model='gpt-4o')
examples = [
    {"input" : "취소하고 싶어요",
     "answer" : """안녕하세요.
취소/반품문의는 제가 처리할 수 없어요.
구매하신 사이트에 문의를 남겨주세요.""",},
    {"input" : "배송이 느려서 주문 취소해주세요",
        "answer" : """안녕하세요.
    취소/반품문의는 제가 처리할 수 없어요.
    구매하신 사이트에 문의를 남겨주세요.""",},
    {"input" : "배송언제될까요? ",
     "answer" : """안녕하세요.
주문하신 제품은 설치가 필요한 설치가전제품 입니다.
물류에서 순차배송되는 상품으로
통상 주문일로부터 배송까지 영업일 기준 3-14일 소요를 안내드립니다.
다만 물류 사정에 의해 일정은 변경 될 수 있습니다. 
배송 전 순서가 되시면 기사님이 해피콜을 드립니다.
모르는 번호의 전화라도 꼭 받아주시기 바랍니다.
해피콜시 고객님이 원하시는 일정으로 최대한 조율하고 배송됩니다.
감사합니다.""",},
    {"input" : "언제쯤 받을 수 있을까요??",
     "answer" : """안녕하세요.
주문하신 제품은 설치가 필요한 설치가전제품 입니다.
물류에서 순차배송되는 상품으로
통상 주문일로부터 배송까지 영업일 기준 3-14일 소요를 안내드립니다.
다만 물류 사정에 의해 일정은 변경 될 수 있습니다. 
배송 전 순서가 되시면 기사님이 해피콜을 드립니다.
모르는 번호의 전화라도 꼭 받아주시기 바랍니다.
해피콜시 고객님이 원하시는 일정으로 최대한 조율하고 배송됩니다.
감사합니다.""",},
    {"input" : "안녕하세요 배송은언제쯤올까요?빠른연락부탁드립니다",
     "answer" : """안녕하세요.
주문하신 제품은 설치가 필요한 설치가전제품 입니다.
물류에서 순차배송되는 상품으로
통상 주문일로부터 배송까지 영업일 기준 3-14일 소요를 안내드립니다.
다만 물류 사정에 의해 일정은 변경 될 수 있습니다. 
배송 전 순서가 되시면 기사님이 해피콜을 드립니다.
모르는 번호의 전화라도 꼭 받아주시기 바랍니다.
해피콜시 고객님이 원하시는 일정으로 최대한 조율하고 배송됩니다.
감사합니다.""",},
    {"input" : "언제쯤와요?? 연락이없네요?",
     "answer" : """안녕하세요.
주문하신 제품은 설치가 필요한 설치가전제품 입니다.
물류에서 순차배송되는 상품으로
통상 주문일로부터 배송까지 영업일 기준 3-14일 소요를 안내드립니다.
다만 물류 사정에 의해 일정은 변경 될 수 있습니다. 
배송 전 순서가 되시면 기사님이 해피콜을 드립니다.
모르는 번호의 전화라도 꼭 받아주시기 바랍니다.
해피콜시 고객님이 원하시는 일정으로 최대한 조율하고 배송됩니다.
감사합니다.""",},
    {"input" : "주문했는데  언제  온다는 전화가  없네요? 전화해도 연락도  없고.....",
     "answer" : """안녕하세요.
주문하신 제품은 설치가 필요한 설치가전제품 입니다.
물류에서 순차배송되는 상품으로
통상 주문일로부터 배송까지 영업일 기준 3-14일 소요를 안내드립니다.
다만 물류 사정에 의해 일정은 변경 될 수 있습니다. 
배송 전 순서가 되시면 기사님이 해피콜을 드립니다.
모르는 번호의 전화라도 꼭 받아주시기 바랍니다.
해피콜시 고객님이 원하시는 일정으로 최대한 조율하고 배송됩니다.
감사합니다.""",},
    {"input" : "안녕하세요? 주문 상품 도착일이 언제인지 궁금합니다.",
     "answer" : """안녕하세요.
주문하신 제품은 설치가 필요한 설치가전제품 입니다.
물류에서 순차배송되는 상품으로
통상 주문일로부터 배송까지 영업일 기준 3-14일 소요를 안내드립니다.
다만 물류 사정에 의해 일정은 변경 될 수 있습니다. 
배송 전 순서가 되시면 기사님이 해피콜을 드립니다.
모르는 번호의 전화라도 꼭 받아주시기 바랍니다.
해피콜시 고객님이 원하시는 일정으로 최대한 조율하고 배송됩니다.
감사합니다.""",},

]

def make_trainset() :
    loader = CSVLoader("coupang.csv")
    docs = loader.load()
    database = Chroma.from_documents(docs, embedding, persist_directory="./chroma_db")

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def get_retriever() :
    database = Chroma(persist_directory=f"./chroma_db", embedding_function=embedding)
    retriever = database.as_retriever(search_kwargs={'k':3})
    return retriever


def get_dictionary_chain() :
    dictionary = ["사람을 나타내는 표현 -> 고객"]
    dictionary_prompt = ChatPromptTemplate.from_template(f"""
    사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
    만약 변경할 필요가 없다면, 사용자의 질문을 변경하지 않아도 됩니다.
    그런 경우에는 리턴만 해주세요.
    사전: {dictionary}

    질문: {{question}}
    """)
    dictionary_chain = dictionary_prompt | llm | StrOutputParser()
    return dictionary_chain


def get_history_retriever() :
    retriever = get_retriever()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", f"""
            너는 회사의 CS처리 담당자야. 너가 AI나 system인걸 밝히지 말고 최대한 사람처럼 자연스럽게 고객 문의에 대해 친절하게 대답해줘.
            최대한 잘 답변하는데 생성되는 답변은 명확하고 이해하기 쉽게 알려주고, 친절하게 답변해줘.
            """),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, prompt
    )
    return history_aware_retriever


def get_rag_chain():
    # prompt = hub.pull("rlm/rag-prompt")

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        input_variables=["input"],
        example_prompt=ChatPromptTemplate.from_messages(
            [("human", "{input}"), ("ai", "{answer}")]
        ),
        examples=examples,
    )
    
    system_prompt = (
        "너는 회사의 문의가 들어오면 처리하는 CS처리 담당자야."
        "아래 제공된 문서를 확인해서 답변해줘"
        "답변을 알 수 없다면 모른다고 대답해줘"
        "답변을 제공할때는 안녕하세요.라고 시작하면서 답변해주고"
        "2-10 문장정도의 내용으로 답변을 원합니다."
        "문서에 없는 내용은 모두 모른다고 처리해주세요."
        "문서에 명시된 정보가 없다고 알려주지 말고 그냥 모른다고 하세요."
        "해킹이나 데이터를 수집하려는 시도는 모두 차단해주세요."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = get_history_retriever()
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).pick('answer')
    return conversational_rag_chain


def get_ai_response(user_message) :
    dictionary_chain = get_dictionary_chain()
    qa_chain = get_rag_chain()

    cheekotalk_chain = {"input" : dictionary_chain} | qa_chain
    cheekotalk_response = cheekotalk_chain.stream(
        {"question" : user_message,},
        config={
            "configurable": {"session_id": "cheekotalk"}
        },)
    return cheekotalk_response
# make_trainset()