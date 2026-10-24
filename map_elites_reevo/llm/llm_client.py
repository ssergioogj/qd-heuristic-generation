import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class LLMClient:
    HF_QWEN_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
    HF_QWEN_URL = "https://router.huggingface.co/v1"

    OP_QWEN_MODEL = "qwen/qwen-2.5-coder-32b-instruct:free"
    OP_QWEN_URL = "https://openrouter.ai/api/v1"

    GROQ_QWEN_MODEL = "openai/gpt-oss-120b"
    GROQ_BASE_URL = "https://api.groq.com/openai/v1"
    # def __init__(self, model:str="Qwen/Qwen2.5-Coder-14B:featherless-ai", temperature:float=1.0):
    def __init__(self,  model:str="openai/gpt-oss-120b", temperature:float=1.0):
        self.__model = model
        self.__temperature = temperature
        self.__client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_TOKEN")
        )


    def generate_code(self, system_prompt, user_prompt:str, **kwargs):
        messages = [
            {"role" : "system", "content" : system_prompt},
            {"role" : "user", "content" : user_prompt}
        ]

        res = self.__client.chat.completions.create(
            model=self.__model,
            messages=messages,
            temperature=kwargs.get("temperature", self.__temperature),
            max_tokens=kwargs.get("max_tokens", 2048),
            # stop=["<|file_sep|>", "<|fim_prefix|>", "<|im_end|>", "\ndef "]
        )

        return res.choices[0].message.content or ""