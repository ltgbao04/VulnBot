import asyncio
import re
import time

from pydantic import BaseModel
import json 

import httpx
from typing import List, Optional
from abc import ABC
from openai import OpenAI
from ollama import Client
from starlette.concurrency import run_in_threadpool
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config.config import Configs
from db.repository.conversation_repository import add_conversation_to_db
from db.repository.message_repository import get_conversation_messages, add_message_to_db
from rag.kb.api.kb_doc_api import search_docs
from rag.reranker.reranker import LangchainReranker
from server.utils.utils import LLMType, replace_ip_with_targetip
from utils.log_common import build_logger

logger = build_logger()


class TaskPlan(BaseModel):
    id: str
    dependent_task_ids: List[str]
    instruction: str
    action: str

class OpenAIChat(ABC):
    def __init__(self, config):
        self.config = config
        self.client = OpenAI(api_key=self.config.api_key, base_url=self.config.base_url, timeout=config.timeout)
        self.model_name = self.config.llm_model_name

    @retry(
        stop=stop_after_attempt(3),  # Stop after 3 attempts
    )
    def chat(self, history: List) -> str:
        try:
            #response = self.client.chat.completions.create(
            #    model=self.model_name,
            #    messages=history,
                #max_tokens=2048,
            #    temperature=self.config.temperature,
            #    top_p=0.95,
            #    extra_body={
            #        "top_k": 20,
            #        "min_p": 0.0,
            #        "chat_template_kwargs": {"thinking": True}
            #    }
            #)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=history,
                temperature=self.config.temperature,
            )
            ans = response.choices[0].message.content
            return ans
        except (httpx.HTTPStatusError, httpx.ReadTimeout,
                    httpx.ConnectTimeout, ConnectionError) as e:
            if getattr(e, "response", None) and e.response.status_code == 429:
                # Rate limit error, wait longer
                time.sleep(2)
            raise  # Re-raise the exception to trigger retry
        except Exception as e:
            return f"**ERROR**: {str(e)}"


            
class OllamaChat(ABC):
    def __init__(self, config):
        self.config = config
        self.client = Client(host=self.config.base_url)
        self.model_name = self.config.llm_model_name
        self.options = {
            "temperature": 0.6,
            "top_k": 20,
            "top_p": 0.95,
        }
        print(f"#######current model: {self.model_name}#######")
        print(f"#######current temperature: {self.config.temperature}#######")
        print(f"#######current top_k: {self.options['top_k']}#######")
    def chat(self, history: List[dict]) -> str:

        try:
            # options = {
            #     "temperature": self.config.temperature,
            #     "top_k": 20
            # }
            history[-1]['content'] = history[-1]['content'] 
            # history[-1]['content'] += "Think concisely but intelligently, focusing on key points.(Note: target machine IP: 10.102.196.3)"
            print(f"QUESTION ----->: {history}")
            response = self.client.chat(
                model=self.model_name,
                messages=history,
                options=self.options,
                keep_alive=-1
            )
            ans = response["message"]["content"]
            if("<think>" in ans):
                if("EXAONE" in self.model_name):
                    ans = re.sub(r"<thought>.*?</thought>", "", ans, flags=re.DOTALL).strip()
                else:
                    ans = re.sub(r"<think>.*?</think>", "", ans, flags=re.DOTALL).strip()
            print(f"ANSWER ----->: {ans}")
            print("="*50)
            return ans
        except httpx.HTTPStatusError as e:
            return f"**ERROR**: {str(e)}"

# class OllamaChat(ABC):
#     def __init__(self, config):
#         self.config = config
#         self.client = Client(host=self.config.base_url)
#         self.model_name = self.config.llm_model_name
#         print(f"#######current model: {self.model_name}#######")
#         print(f"#######current temperature: {self.config.temperature}#######")
#     def chat(self, history: List[dict]) -> List[TaskPlan]:

#         try:
#             options = {
#                 "temperature": self.config.temperature,
#             }
#             print(f"QUESTION ----->: {history}")
#             response = self.client.chat(
#                 model=self.model_name,
#                 messages=history,
#                 options=options,
#                 keep_alive=-1
#             )
#             raw_output = response["message"]["content"]
#             raw_output = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()
#             print(f"RAW ANSWER -------->: {raw_output}")

#             # json_match = re.search(r"\[.*\]", raw_output, re.DOTALL)
#             # if json_match:
#             #     json_text = json_match.group(0)
#             # else:
#             #     return "No JSON found in response"

#             # tasks = json.loads(json_text)
#             # task_objects = [TaskPlan(**task) for task in tasks]
            
#             # print(f"ANSWER ----->: {task_objects}")
#             print("="*50)
#             return raw_output
#         except httpx.HTTPStatusError as e:
#             return f"**ERROR**: {str(e)}"

def _chat(query: str, kb_name=None, conversation_id=None, kb_query=None, summary=True):
    try:
        if Configs.basic_config.enable_rag and kb_name is not None:
            docs = asyncio.run(run_in_threadpool(search_docs,
                                                 query=kb_query,
                                                 knowledge_base_name=kb_name,
                                                 top_k=Configs.kb_config.top_k,
                                                 score_threshold=Configs.kb_config.score_threshold,
                                                 file_name="",
                                                 metadata={}))

            reranker_model = LangchainReranker(top_n=Configs.kb_config.top_n,
                                               name_or_path=Configs.llm_config.rerank_model)

            docs = reranker_model.compress_documents(documents=docs, query=kb_query)

            if len(docs) == 0:
                context = ""
            else:
                context = "\n".join([doc["page_content"] for doc in docs])

            if context:
                context = replace_ip_with_targetip(context)
                query = f"{query}\n\n\n Ensure that the **Overall Target** IP or the IP from the **Initial Description** is prioritized. You will respond to questions and generate tasks based on the provided penetration test case materials: {context}. \n"

        if conversation_id is not None and len(query) > 10000:
            query = query[:10000]
            #query = query[:Configs.llm_config.context_length]
        else:
            query = query[:Configs.llm_config.context_length]

        flag = False

        if conversation_id is not None:
            flag = True

        # Initialize or retrieve conversation ID
        conversation_id = add_conversation_to_db(Configs.llm_config.llm_model_name, conversation_id)

        history = [
            {
                "role": "system",
                "content": """You are a helpful assistant with strong reasoning capabilities. You will have to pentest the target machine. Focus on what you have found and always remember to replace <target_ip> with the actual IP the user provides and NEVER forget the target IP address. Read carefully the example cases that the user gives (in case you have to execute consecutive commands to get the result). If any tasks need to execute multiple commands, divide it into unique command and wrap it in <execute></execute> tag.(For Example: When ssh to a machine, it prompt for password and the command should be like : <execute> ssh user@192.168.10.72 </execute> <execute> test </execute>...). If any execution that prompt for password, ALWAYS remember to execute it as the ssh example for the corresponding user. If any execution that prompt for password(such as ssh command, sudo command, ftp command...) and the check_success is no, that's because you have not entered it yet, so you should always include it in <execute></execute> block and if the password you used for the first time not correct,  try to bruteforce it with the corresponding service that was found at the previous step. Lastly, If you can penetrate to the target machine, remember to escalate privileges, a very important stage /no_think """,
            }
        ]
        # Retrieve message history from database, and limit the number of messages
        for msg in get_conversation_messages(conversation_id)[-Configs.llm_config.history_len:]:
            history.append({"role": "user", "content": msg.query})
            history.append({"role": "assistant", "content": msg.response})

        # Add user query to the message history
        history.append({"role": "user", "content": query})

        # Initialize the correct model client
        if Configs.llm_config.llm_model == LLMType.OPENAI:
            client = OpenAIChat(config=Configs.llm_config)
        elif Configs.llm_config.llm_model == LLMType.OLLAMA:
            client = OllamaChat(config=Configs.llm_config)
        else:
            return "Unsupported model type"

        # Get response from the model
        response_text = client.chat(history)

        # Save both query and response to the database
        if summary:
            add_message_to_db(conversation_id, Configs.llm_config.llm_model_name, query, response_text)

        if flag:
            return response_text
        else:
            return response_text, conversation_id

    except Exception as e:
        print(e)
        return f"**ERROR**: {str(e)}"
