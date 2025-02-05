# components/llm_agent.py

import anthropic

GENERAL_ROLE = """You are an AI Chatbot dedicated to supporting Switzerland's solar energy project...
(unchanged system prompt text here)
...Keep answers concise and avoid bullet points."""

class ClaudeAgent:
    def __init__(self, client):
        self.client = client

    def generate_prompt(self, conversation_messages, rag_context):
        """
        Builds the conversation text to be sent as a user message.
        """
        conversation_text = ""
        for msg in conversation_messages:
            if msg["role"] == "assistant":
                conversation_text += f"Assistant: {msg['content']}\n\n"
            else:
                conversation_text += f"User: {msg['content']}\n\n"

        user_msg = (
            f"{conversation_text}"
            f"(Use the following context if relevant):\n{rag_context}\n\n"
        )
        return user_msg

    def call_claude(self, conversation_messages, rag_context, max_tokens=700, temperature=0.7):
        """
        Calls Claude with the entire conversation + new context, returns the assistant's reply.
        """
        user_msg = self.generate_prompt(conversation_messages, rag_context)

        response = self.client.messages.create(
            model="claude-3-5-haiku-20241022",  # Replace with your chosen Claude model
            max_tokens=max_tokens,
            temperature=temperature,
            system=GENERAL_ROLE,  # Pass system prompt separately
            messages=[
                {"role": "user", "content": user_msg}
            ]
        )
        return response.content[0].text.strip()

    def summarize_text(self, text, max_tokens=1024, temperature=0.3):
        """
        Produces a concise summary of the given text using Claude.
        Uses message-based format to ensure the text is fully included.
        """
        # Fallback if there's no text
        if not text.strip():
            return "No text to summarize."

        system_msg = (
            "You are a helpful AI that summarizes text into a concise paragraph. "
            "Focus on the key insights. Avoid excessive details or bullet points."
        )
        user_msg = f"Please summarize the following text:\n\n{text}"

        response = self.client.messages.create(
            model="claude-3-5-haiku-20241022",    # Or "claude-2-100k", "claude-1.3", etc.
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_msg,  # Pass system prompt separately
            messages=[
                {"role": "user", "content": user_msg}
            ]
        )
        return response.content[0].text.strip()
