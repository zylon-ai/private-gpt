import os

from dotenv import load_dotenv
load_dotenv()

params = {
    'translator': os.environ.get('TRANSLATE_ENGINE',"GoogleTranslator"), # GoogleTranslator or OneRingTranslator.
    'custom_url': os.environ.get('TRANSLATE_CUSTOM_URL',"http://127.0.0.1:4990/"), # custom url for OneRingTranslator server
    'user_lang': os.environ.get('TRANSLATE_USER_LANG','en'), # user language two-letters code like "fr", "es" etc. "en" for NO translation
    'translate_user_input': (os.environ.get('TRANSLATE_USER_INPUT',"0") == "1"), # translate user input to EN
    'translate_system_output': (os.environ.get('TRANSLATE_SYSTEM_OUTPUT',"0") == "1"), # translate system output to UserLang
}
#print(params)
def translator_main(string,from_lang:str,to_lang:str) -> str:
    if from_lang == to_lang: return string

    from deep_translator import GoogleTranslator
    res = ""
    if params['translator'] == "GoogleTranslator":
        res = GoogleTranslator(source=from_lang, target=to_lang).translate(string)
    if params['translator'] == "OneRingTranslator":
        #print("GoogleTranslator using")
        #return GoogleTranslator(source=params['language string'], target='en').translate(string)
        custom_url = params['custom_url']
        if custom_url == "":
            res = "Please, setup custom_url for OneRingTranslator (usually http://127.0.0.1:4990/)"
        else:
            import requests
            response_orig = requests.get(f"{custom_url}translate", params={"text":string,"from_lang":from_lang,"to_lang":to_lang})
            if response_orig.status_code == 200:
                response = response_orig.json()
                #print("OneRingTranslator result:",response)

                if response.get("error") is not None:
                    print(response)
                    res = "ERROR: "+response.get("error")
                elif response.get("result") is not None:
                    res = response.get("result")
                else:
                    print(response)
                    res = "Unknown result from OneRingTranslator"
            elif response_orig.status_code == 404:
                res = "404 error: can't find endpoint"
            elif response_orig.status_code == 500:
                res = "500 error: OneRingTranslator server error"
            else:
                res = f"{response_orig.status_code} error"

    return res

from typing import Any

from langchain.prompts import PromptTemplate
class PromptTemplateTrans(PromptTemplate):
    def format(self, **kwargs: Any) -> str:
        res = super().format(**kwargs)
        if params["translate_user_input"]:
            res = translator_main(res,params["user_lang"],"en")
            #print("Translated prompt: ",res)
        return res
