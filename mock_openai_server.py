#!/usr/bin/env python3
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


EXAMPLE_ANSWERS = {
    "What is the capital city of Australia?": "Canberra",
    "What is the chemical symbol for the element gold?": "Au",
    "In which year did the first crewed Moon landing take place?": "1969",
    "Which ocean is the largest by surface area?": "The Pacific Ocean",
}


def extract_question(messages):
    for msg in reversed(messages or []):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if content:
                return content.strip()
    return ""


def build_answer(question):
    answer = EXAMPLE_ANSWERS.get(question)
    if answer:
        return f"<answer>{answer}</answer>"

    lowered = question.lower()
    if "capital" in lowered and "australia" in lowered:
        return "<answer>Canberra</answer>"
    if "chemical symbol" in lowered and "gold" in lowered:
        return "<answer>Au</answer>"
    if "moon landing" in lowered:
        return "<answer>1969</answer>"
    if "largest" in lowered and "ocean" in lowered:
        return "<answer>The Pacific Ocean</answer>"

    short = re.sub(r"\s+", " ", question).strip()
    return f"<answer>Mock answer for: {short[:120]}</answer>"


class Handler(BaseHTTPRequestHandler):
    server_version = "MockOpenAI/0.1"

    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return

    def do_GET(self):
        if self.path in ("/health", "/v1/models"):
            if self.path == "/health":
                return self._send_json({"status": "ok"})
            return self._send_json(
                {
                    "object": "list",
                    "data": [{"id": "mock-searchswarm", "object": "model"}],
                }
            )
        self._send_json({"error": "not found"}, status=404)

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            return self._send_json({"error": "not found"}, status=404)

        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            return self._send_json({"error": "invalid json"}, status=400)

        messages = payload.get("messages", [])
        question = extract_question(messages)
        answer_text = build_answer(question)

        response = {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": 0,
            "model": payload.get("model", "mock-searchswarm"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": answer_text,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": max(1, len(question.split())),
                "completion_tokens": max(1, len(answer_text.split())),
                "total_tokens": max(2, len(question.split()) + len(answer_text.split())),
            },
        }
        self._send_json(response)


def main():
    host = os.environ.get("MOCK_OPENAI_HOST", "127.0.0.1")
    port = int(os.environ.get("MOCK_OPENAI_PORT", "18080"))
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Mock OpenAI server listening on http://{host}:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
