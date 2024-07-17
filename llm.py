from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from config import answer_examples


store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def get_retriever():
    embedding = OpenAIEmbeddings(model='text-embedding-3-large')
    index_name = 'traffic-markdown-index'
    database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)
    retriever = database.as_retriever(search_kwargs={'k': 4})
    return retriever


def get_history_retriever():
    llm = get_llm()
    retriever = get_retriever()

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    return history_aware_retriever


def get_llm(model='gpt-4o'):
    llm = ChatOpenAI(model=model)
    return llm


def get_dictionary_chain():
    dictionary = [
        "교통법규 위반을 나타내는 표현 -> 교통법규 위반",
        "교통사고 피해자를 나타내는 표현 -> 피해자",
        "교통법규 위반에 대한 처벌을 나타내는 표현 -> 처벌",
        "공동 운전자를 나타내는 표현 -> 공동 운전자",
        "고의적 교통법규 위반을 나타내는 표현 -> 고의",
        "과실로 인한 교통법규 위반을 나타내는 표현 -> 과실",
        "운전자의 변호권을 나타내는 표현 -> 변호권",
        "운전면허 정지 및 취소를 나타내는 표현 -> 면허 정지/취소",
        "교통사고 재판을 나타내는 표현 -> 재판",
        "교통법규 위반에 대한 형 집행을 나타내는 표현 -> 형 집행",
        "교통사고 피해자에 대한 보호관찰을 나타내는 표현 -> 보호관찰",
        "운전자의 자백을 나타내는 표현 -> 자백",
        "교통사고 증거를 나타내는 표현 -> 증거",
        "교통법규 위반에 대한 고소를 나타내는 표현 -> 고소",
        "교통법규 위반에 대한 기소를 나타내는 표현 -> 기소",
        "불법 주/정차를 나타내는 표현 -> 불법 주/정차",
        "교통법규 위반에 대한 유죄 판결을 나타내는 표현 -> 유죄",
        "교통법규 위반에 대한 무죄 판결을 나타내는 표현 -> 무죄",
        "교통사고 혐의를 나타내는 표현 -> 혐의",
        "교통법규 위반에 대한 집행유예를 나타내는 표현 -> 집행유예"
    ]
    llm = get_llm()

    prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        그런 경우에는 질문만 리턴해주세요.
        사전: {dictionary}

        질문: {{question}}
    """)

    dictionary_chain = prompt | llm | StrOutputParser()
    return dictionary_chain


def get_rag_chain():
    llm = get_llm()

    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{answer}"),
        ]
    )
    # few_shot_prompt = FewShotChatMessagePromptTemplate(
    #     example_prompt=example_prompt,
    #     examples=answer_examples,
    # )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=answer_examples,
        input_variables=['input']
    )
    system_prompt = (
        "당신은 도로교통법 전문가입니다. 사용자의 도로교통법에 관한 질문에 답변해 주세요. "
        "아래에 제공된 문서를 활용해서 답변해 주시고, "
        "답변을 알 수 없다면 모른다고 답변해 주세요. "
        "답변을 제공할 때는 '도로교통법 (XX조)에 따르면' 이라고 시작하면서 답변해 주시고, "
        # "2-3 문장 정도의 짧은 내용의 답변을 원합니다."
        "사용자가 명쾌하게 이해할 수 있는 내용의 답변을 원합니다. "
        "답변의 내용에 '처벌', '구류', '벌금', '과태료'에 관한 내용도 추가해서 답변해 주세요. "
        "ChatGPT 보다 나은 답변이 나온다면 당신은 두둑한 보너스를 받게 됩니다."
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


def get_ai_response(user_message):
    dictionary_chain = get_dictionary_chain()
    rag_chain = get_rag_chain()
    tax_chain = {"input": dictionary_chain} | rag_chain
    ai_response = tax_chain.stream(
        {
            "question": user_message
        },
        config=
        {
            "configurable": {"session_id": "abc123"}
        },
    )

    return ai_response
