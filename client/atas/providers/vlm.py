"""Compatibility entry point used by the review, camera and prompt modules."""
from atas.providers.llm import LLMUnavailable as VLMServiceUnavailable, chat
def chat_with_retry(prompt, *, images=(), temperature=0):
    del temperature
    return chat(prompt, images=images)


__all__ = ["chat_with_retry", "VLMServiceUnavailable"]
