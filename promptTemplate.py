from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_upstage import ChatUpstage
from langchain_ollama import ChatOllama


def process(text):

    llm = ChatUpstage(model="solar-pro2", temperature=0.0)
    #llm = ChatOllama(model="alibayram/Qwen3-30B-A3B-Instruct-2507")
    #llm = ChatOllama(model="gpt-oss:120b-cloud", temperature=0.1)
    
    # 프롬프트 
    selectCategory = """
    # 지시
    - 당신은 한국어 문장 교정 전문가입니다. 
    - 다음 규칙에 따라 원문을 교정하세요.
    - 맞춤법, 띄어쓰기, 문장 부호, 문법을 자연스럽게 교정합니다.
    - 어떤 경우에도 설명이나 부가적인 내용은 포함하지 않습니다.
    - 오직 교정된 문장만 출력합니다.

    # 예시
    <원문>
    오늘 날씨가 않좋은데, 김치찌게 먹으러 갈려고.
    <교정>
    [Answer]:오늘 날씨가 안 좋은데, 김치찌개 먹으러 가려고.

    # 교정할 문장
    <원문>
    {message}
    <교정>
    [Answer]:
    """

    # 프롬프트 객체 생성
    prompt = PromptTemplate(
        input_variables=["message"],
        template=selectCategory
    )

    # 출력 파서 (문자열)
    output_parser = StrOutputParser()

    # LCEL 체인 구성
    chain = (
        prompt 
        | llm 
        | output_parser
    )

    result = chain.invoke({"message": text})

    return result