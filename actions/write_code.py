from pydantic import BaseModel, Field
from actions.execute_task import ExecuteTask
from prompts.prompt import DeepPentestPrompt
from server.chat.chat import _chat_function_calling
from utils.log_common import build_logger
import re

logger = build_logger()

class WriteCode(BaseModel):
    next_task: str
    action: str

    def run(self):
        # Use next_task directly as the query (next_task đã chứa instruction từ planner)
        instruction = self.next_task
        logger.info(f"Using instruction: {instruction[:100]}... (truncated)")

        query = instruction
        logger.info(f"Query: {query[:100]}... (truncated)")

        # Call _chat_function_calling
        logger.info("Calling _chat_function_calling...")
        response = _chat_function_calling(query=query)
        logger.info(f"LLM Response: {response}")

        # Extract only the response_text from the tuple (ignoring conversation_id)
        instruction = response[0] if isinstance(response, tuple) else response
        logger.info(f"Extracted instruction: {instruction}")

        # Check if instruction contains a valid <execute> command or tool call
        execute_matches = re.findall(r'<execute>(.*?)</execute>', instruction, re.DOTALL)
        if execute_matches:
            code = [cmd.strip() for cmd in execute_matches]
            logger.info(f"Extracted commands: {code}")
        else:
            logger.warning("No valid <execute> command found in instruction, skipping execution")
            code = []

        # Execute the generated command
        logger.info("Executing task with ExecuteTask...")
        code_executor = ExecuteTask(action=self.action, instruction=instruction, code=code)
        result = code_executor.run()
        logger.info(f"Execution result: {result}")

        return result