import anthropic

GENERAL_ROLE = """You are an AI Chatbot dedicated to supporting Switzerland's solar energy project...
(unchanged system prompt text here)
...Keep answers concise and avoid bullet points."""

class ClaudeAgent:
    def __init__(self, client):
        self.client = client

    def generate_prompt(self, conversation_messages, rag_context):
        """
        Build the system + conversation + user question into a single prompt.
        """
        conversation_text = ""
        for msg in conversation_messages:
            if msg["role"] == "assistant":
                conversation_text += f"Assistant: {msg['content']}\n\n"
            else:
                conversation_text += f"User: {msg['content']}\n\n"

        prompt = (
            f"{anthropic.HUMAN_PROMPT}SYSTEM: {GENERAL_ROLE}\n\n"
            f"{conversation_text}"
            f"(Use the following context if relevant):\n{rag_context}\n\n"
            f"{anthropic.AI_PROMPT}"
        )
        return prompt

    def call_claude(self, conversation_messages, rag_context, max_tokens=700, temperature=0.7):
        """
        Calls Claude with the entire conversation + new context, returns the assistant's reply.
        """
        prompt = self.generate_prompt(conversation_messages, rag_context)
        response = self.client.messages.create(
            model="claude-3-5-haiku-20241022",  # or "claude-2.0-100k", etc. 
                               #  (use your desired Claude variant)
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
