from langchain import LlamaCpp
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseOutputParser
import numpy as np
import pandas as pd


class Collapser(BaseOutputParser[str]):
    """Parse the output of an LLM"""

    def parse(self, text: str) -> str:
        """Parse the output of an LLM call. Collapse enter to 'yes', 'no' or 'NA'"""

        text = text.split("\n")[0].lower()
        text_processed = text.replace(" ", "").replace(".", "")
        if text_processed not in ["yes", "no"]:
            text_processed = text.split(" ")
            for elt in text_processed:
                if elt in ["yes", "no"]:
                    return elt
            return "NA"
        else:
            return text_processed


class Extraction(BaseOutputParser[str]):
    """Parse the output of an LLM"""

    def parse(self, text: str) -> str:
        """ """
        text = text.split("\n")[0].replace(" ", "").replace(".", "")
        text_processed = text.split("=")
        if len(text_processed) > 1:
            print(text_processed)
            text = text_processed[1]
        return text


def craft_spot_chain(model_path: str):
    """craft a chain to ..."""

    # load model
    lllm = LlamaCpp(model_path=model_path, n_gpu_layers=30)

    # define chat prompt
    template = "You are helpful assistant that read a text and answer questions about what you read. return only 'Yes' or 'No, nothing else.TEXT:{text}. QUESTION: does the text contains information about {target} ? ANSWER:"

    # assemble chain
    chat_prompt = ChatPromptTemplate.from_messages([("system", template)])

    # craft chain
    chain = chat_prompt | lllm | Collapser()

    # return chain
    return chain


def craft_extract_chain(model_path: str):
    """craft a chain to ..."""

    # load model
    lllm = LlamaCpp(model_path=model_path, n_gpu_layers=30)

    # define chat prompt
    template = "You are helpful assistant that read a text and extract the value of the parameter {target}. return only the value of {target}, nothing else.TEXT:{text}. VALUE :"

    # assemble chain
    chat_prompt = ChatPromptTemplate.from_messages([("system", template)])

    # craft chain
    chain = chat_prompt | lllm | Extraction()

    # return chain
    return chain


def extract(text_list, target_list, model_path):
    """ """

    # craft spot chain
    spotter = craft_spot_chain(model_path)
    extractor = craft_extract_chain(model_path)

    # loop over texts
    data = []
    for text in text_list:

        # init dict
        target_to_value = {}
        target_to_value["text"] = text

        # loop over targets
        for target in target_list:

            # look for presence of the target in the text
            presence = spotter.invoke({"text": text, "target": target})

            # if target is present, go for extraction
            if presence == "yes":
                value = extractor.invoke({"text": text, "target": target})
                target_to_value[target] = value
            else:
                target_to_value[target] = np.nan

        # udpate list
        data.append(target_to_value)

    # craft and return dataframe
    return pd.DataFrame(data)


if __name__ == "__main__":

    # parameters
    model_path = "/home/bran/Workspace/misc/llama/models/llama-2-13b-chat.Q5_K_M.gguf"
    text_list = [
        "Jimmy est un perroquet qui mesure 65 cm et il pèse environ 250kg",
        "Robert est un saint-bernard qui as une taille de 1 mètre et un poid de 25 kilos",
    ]
    # target = "Color"
    #
    # # test
    # chain = craft_spot_chain(model_path)
    # answer = chain.invoke({"text": text, "target": target})
    # print(answer)

    d = extract(text_list, ["Size", "Color", "Name", "Weight"], model_path)
    print(d)
