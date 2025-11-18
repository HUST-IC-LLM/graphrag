import json
import logging
import time
from typing import Any, Callable, Dict, Generator, Optional

import requests


class HiAgentLLMClient:
    app_conversation_id: str
    user_id: str

    def __init__(
        self,
        api_url: str,
        api_key: str,
        user_agent: str,
    ) -> None:
        self.api_url: str = api_url
        self.api_key: str = api_key
        self.user_agent: str = user_agent
        self.session: requests.Session = requests.Session()
        self.session.headers.update(
            {
                "Apikey": api_key,
                "Content-Type": "application/json",
                "User-Agent": user_agent,
            }
        )

    def create_conversation(
        self,
        user_id: str,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = f"{self.api_url}/api/proxy/api/v1/create_conversation"
        payload = {
            "AppKey": self.api_key,
            "Inputs": inputs or {},
            "UserID": user_id,
        }
        try:
            response = self.session.post(url, json=payload, timeout=10)
            response.raise_for_status()
            self.app_conversation_id = (
                response.json().get("Conversation").get("AppConversationID")
            )
            self.user_id = user_id
            return response.json()
        except requests.RequestException as e:
            logging.error(f"Error creating conversation: {e}")
            raise

    def chat_query_blocking(
        self,
        query: str,
        query_extends: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any] | Any:
        url = f"{self.api_url}/api/proxy/api/v1/chat_query_v2"
        payload = {
            "Query": query,
            "AppConversationID": self.app_conversation_id,
            "AppKey": self.api_key,
            "ResponseMode": "blocking",
            "UserID": self.user_id,
            "QueryExtends": query_extends,
        }
        try:
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logging.error(f"Error during chat query: {e}")
            raise

    def chat_query_streaming(
        self,
        query: str,
        query_extends: Optional[Dict[str, Any]] = None,
        on_message: Optional[Callable] = None,
        chunk_size: int = 1,
        debug: bool = False,
    ) -> Generator[Dict[str, Any], None, None]:
        url = f"{self.api_url}/api/proxy/api/v1/chat_query_v2"
        payload = {
            "Query": query,
            "AppConversationID": self.app_conversation_id,
            "AppKey": self.api_key,
            "ResponseMode": "streaming",
            "UserID": self.user_id,
            "QueryExtends": query_extends,
        }

        try:
            start_time = time.time()
            response = self.session.post(
                url,
                json=payload,
                stream=True,
                timeout=30,
                headers={
                    "Connection": "keep-alive",
                    "Accept": "text/event-stream",
                    "Cache-Control": "no-cache",
                },
            )
            response.raise_for_status()

            buffer = b""
            event_count = 0
            last_event_time = start_time

            for chunk in response.iter_content(
                chunk_size=chunk_size, decode_unicode=False
            ):
                if not chunk:
                    continue

                if debug:
                    current_time = time.time()
                    time_since_last = current_time - last_event_time
                    print(
                        f"\n[DEBUG] Received chunk: {len(chunk)} bytes, "
                        f"time since last: {time_since_last:.3f}s"
                    )
                    last_event_time = current_time

                buffer += chunk

                while b"\n" in buffer:
                    line_bytes, buffer = buffer.split(b"\n", 1)

                    try:
                        line = line_bytes.decode("utf-8").strip()
                    except UnicodeDecodeError:
                        if debug:
                            print("[DEBUG] UnicodeDecodeError, continuing...")
                        continue

                    if not line:
                        continue

                    if line.startswith("data:"):
                        data_str = line[5:].strip()
                        try:
                            data = json.loads(data_str)
                            event_count += 1

                            if debug:
                                answer = data.get("answer", "")
                                print(
                                    f"[DEBUG] Event #{event_count}: "
                                    f"type={data.get('event')}, "
                                    f"answer_len={len(answer)}, "
                                    f"content='{answer}'"
                                )

                            if on_message:
                                on_message(data)
                            yield data

                        except json.JSONDecodeError:
                            if debug:
                                print(f"[DEBUG] JSONDecodeError: {data_str[:100]}...")
                            continue

            if buffer:
                try:
                    line = buffer.decode("utf-8").strip()
                    if line.startswith("data:"):
                        data_str = line[5:].strip()
                        data = json.loads(data_str)
                        if on_message:
                            on_message(data)
                        yield data
                except (UnicodeDecodeError, json.JSONDecodeError):
                    pass

            if debug:
                total_time = time.time() - start_time
                print(
                    f"\n[DEBUG] Stream completed: {event_count} events, "
                    f"total time: {total_time:.3f}s"
                )

        except requests.RequestException as e:
            logging.error(f"Error during streaming chat query: {e}")
            raise

    def get_conversation_messages(self, limit: int) -> Dict[str, Any]:
        url = f"{self.api_url}/api/proxy/api/v1/get_conversation_messages"
        payload = {
            "AppKey": self.api_key,
            "UserID": self.user_id,
            "AppConversationID": self.app_conversation_id,
            "Limit": limit,
        }
        try:
            response = self.session.post(url, json=payload, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logging.error(f"Error getting conversation messages: {e}")
            raise

    def get_message_info(self, message_id: str) -> Dict[str, Any]:
        url = f"{self.api_url}/api/proxy/api/v1/get_message_info"
        payload = {
            "AppKey": self.api_key,
            "UserID": self.user_id,
            "MessageID": message_id,
        }
        try:
            response = self.session.post(url, json=payload, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logging.error(f"Error getting message info: {e}")
            raise

    # TODO: TO BE TESTED
    def run_app_workflow(self, input: Dict[str, Any], app_id: str) -> Dict[str, Any]:
        url = f"{self.api_url}/api/proxy/api/v1/run_app_workflow"
        payload = {
            "AppKey": self.api_key,
            "UserID": self.user_id,
            "Input": input,
            "AppID": app_id,
        }
        try:
            response = self.session.post(url, json=payload, timeout=10)
            response.raise_for_status()
            return response.json()

        except requests.RequestException as e:
            logging.error(f"Error running app workflow: {e}")
            raise

    # TODO: TO BE TESTED
    def query_run_app_process(
        self, input: Dict[str, Any], app_id: str
    ) -> Dict[str, Any]:
        url = f"{self.api_url}/api/proxy/api/v1/query_run_app_process"
        payload = {
            "AppKey": self.api_key,
            "UserID": self.user_id,
            "Input": input,
            "AppID": app_id,
        }
        try:
            response = self.session.post(url, json=payload, timeout=10)
            response.raise_for_status()
            return response.json()

        except requests.RequestException as e:
            logging.error(f"Error running app workflow: {e}")
            raise
