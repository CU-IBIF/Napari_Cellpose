"""Shared helpers for tests that log in to a throwaway OpenSSH server."""
import socket
import time
from pathlib import Path

TESTS = Path(__file__).parent
FAKE_SLURM = TESTS / "fake_slurm"


class Prompts:
    """Answers prompts automatically and records them."""

    def __init__(self, answers=(), confirm=True):
        self.answers, self.confirm_answer = list(answers), confirm
        self.asked, self.confirmed = [], []

    def ask(self, title, instructions, prompts):
        self.asked.append((title, instructions, [p for p, _ in prompts]))
        return [self.answers.pop(0) for _ in prompts]

    def confirm(self, title, text):
        self.confirmed.append(title)
        return self.confirm_answer


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_messages(gen, messages, timeout=120):
    start = time.monotonic()
    try:
        while True:
            m = next(gen)
            if m:
                messages.append(m)
            if time.monotonic() - start > timeout:
                raise TimeoutError(messages)
    except StopIteration as stop:
        return stop.value
