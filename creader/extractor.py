from langchain import LlamaCpp
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseOutputParser


def craft_chain(model_path: str):
    """craft a chain to ..."""

    # load model
    lllm = LlamaCpp(model_path=model_path)

    # define chat prompt
    template = "You are helpful assistant that read a text and answer questions about what you read. return only 'Yes' or 'No, nothing else.TEXT:{text}. QUESTION: does the text contains information about {target} ? ANSWER:"

    # assemble chain
    chat_prompt = ChatPromptTemplate.from_messages([("system", template)])

    # craft chain
    chain = chat_prompt | lllm

    # return chain
    return chain


def parse_output():
    """ """


def extractor():
    """ """

    # craft preprompt

    # craft chain

    # invoke chain


if __name__ == "__main__":

    # parameters
    model_path = "/home/bran/Workspace/misc/llama/models/llama-2-13b-chat.Q5_K_M.gguf"
    text = "Jimmy est un perroquet qui mesure 65 cm"
    target = "Size"

    # test
    chain = craft_chain(model_path)
    answer = chain.invoke({"text": text, "target": target})
    print(answer)
