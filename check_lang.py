from dotenv import load_dotenv
import os
from deep_translator import GoogleTranslator
import langdetect
load_dotenv()

auto_translate = os.environ.get("AUTO_TRANSLATE")

def translate(text):
    if auto_translate == None or auto_translate == "false" or auto_translate == "False" or auto_translate == "0":
        return text
    else:
        if langdetect.detect(text) == "en":
            return text
        new_text =  GoogleTranslator(source="auto", target="en").translate(text)
        print(f"Translated '{text}' to '{new_text}'")
        return new_text

if __name__ == "__main__":
    print(translate("Qual è la massa di un elettrone?"))