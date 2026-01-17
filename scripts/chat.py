#!/usr/bin/env python3
"""Interactive chat CLI for Engram."""

import asyncio
import sys

import httpx

API_BASE = "http://localhost:8000"

HELP_TEXT = """
Engram Interactive Chat
=======================

Commands:
  /stats      - Show system statistics
  /concepts   - List top concepts
  /memories   - List recent memories
  /episodes   - List recent episodes
  /feedback + - Give positive feedback to last answer
  /feedback - - Give negative feedback to last answer
  /help       - Show this help
  /quit       - Exit

Just type your question to chat!
"""


class EngramChat:
    def __init__(self):
        self.client = httpx.AsyncClient(base_url=API_BASE, timeout=120.0)
        self.last_episode_id: str | None = None

    async def close(self):
        await self.client.aclose()

    async def chat(self, message: str) -> str:
        """Send a message and get a response."""
        try:
            response = await self.client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": message}]},
            )
            response.raise_for_status()
            data = response.json()

            self.last_episode_id = data.get("episode_id")
            answer = data["choices"][0]["message"]["content"]
            confidence = data.get("confidence", 0)
            concepts = data.get("concepts_activated", [])
            memories = data.get("memories_used", 0)

            result = f"\n{answer}\n"
            result += f"\n[confidence: {confidence:.0%} | concepts: {len(concepts)} | memories: {memories}]"
            if self.last_episode_id:
                result += f"\n[episode: {self.last_episode_id[:20]}...]"

            return result

        except httpx.HTTPError as e:
            return f"Error: {e}"

    async def feedback(self, positive: bool) -> str:
        """Give feedback on the last answer."""
        if not self.last_episode_id:
            return "No previous answer to give feedback on."

        try:
            response = await self.client.post(
                "/v1/feedback",
                json={
                    "episode_id": self.last_episode_id,
                    "feedback": "positive" if positive else "negative",
                },
            )
            response.raise_for_status()
            data = response.json()

            if positive:
                return (
                    f"Positive feedback recorded.\n"
                    f"  Memories strengthened: {data.get('memories_strengthened', 0)}\n"
                    f"  Concepts strengthened: {data.get('concepts_strengthened', 0)}"
                )
            else:
                alt = data.get("alternative_answer")
                result = f"Negative feedback recorded.\n  Memories weakened: {data.get('memories_weakened', 0)}"
                if alt:
                    result += f"\n\nAlternative answer:\n{alt}"
                return result

        except httpx.HTTPError as e:
            return f"Error: {e}"

    async def stats(self) -> str:
        """Get system statistics."""
        try:
            response = await self.client.get("/admin/stats")
            response.raise_for_status()
            data = response.json()

            return (
                f"System Statistics:\n"
                f"  Concepts:         {data['concepts_count']}\n"
                f"  Semantic Memories: {data['semantic_memories_count']}\n"
                f"  Episodic Memories: {data['episodic_memories_count']}\n"
                f"  Documents:        {data['documents_count']}"
            )

        except httpx.HTTPError as e:
            return f"Error: {e}"

    async def concepts(self, limit: int = 20) -> str:
        """List top concepts."""
        try:
            response = await self.client.get(f"/admin/concepts?limit={limit}")
            response.raise_for_status()
            data = response.json()

            lines = [f"Top {len(data)} Concepts (by activation):"]
            for c in data:
                lines.append(f"  {c['name']:30} [{c['type']:10}] activations: {c['activation_count']}")

            return "\n".join(lines)

        except httpx.HTTPError as e:
            return f"Error: {e}"

    async def memories(self, limit: int = 10) -> str:
        """List recent memories."""
        try:
            response = await self.client.get(f"/admin/memories?limit={limit}")
            response.raise_for_status()
            data = response.json()

            lines = [f"Top {len(data)} Memories (by importance):"]
            for m in data:
                content = m['content'][:60] + "..." if len(m['content']) > 60 else m['content']
                lines.append(f"  [{m['memory_type']:10}] imp:{m['importance']:.1f} str:{m['strength']:.2f} | {content}")

            return "\n".join(lines)

        except httpx.HTTPError as e:
            return f"Error: {e}"

    async def episodes(self, limit: int = 10) -> str:
        """List recent episodes."""
        try:
            response = await self.client.get(f"/admin/episodes?limit={limit}")
            response.raise_for_status()
            data = response.json()

            lines = [f"Recent {len(data)} Episodes:"]
            for e in data:
                query = e['query'][:40] + "..." if len(e['query']) > 40 else e['query']
                feedback = e['feedback'] or "none"
                lines.append(
                    f"  [{feedback:10}] +{e['success_count']}/-{e['failure_count']} "
                    f"| {e['behavior_name'][:20]:20} | {query}"
                )

            return "\n".join(lines)

        except httpx.HTTPError as e:
            return f"Error: {e}"


async def main():
    print(HELP_TEXT)

    chat = EngramChat()

    # Check connection
    try:
        await chat.client.get("/health")
        print("Connected to Engram API at", API_BASE)
    except httpx.HTTPError:
        print(f"Error: Cannot connect to Engram API at {API_BASE}")
        print("Make sure the API is running: python -m engram.api.main")
        return

    print("-" * 50)

    try:
        while True:
            try:
                user_input = input("\nYou: ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ["/quit", "/exit", "/q"]:
                print("Goodbye!")
                break

            elif user_input.lower() == "/help":
                print(HELP_TEXT)

            elif user_input.lower() == "/stats":
                result = await chat.stats()
                print(f"\n{result}")

            elif user_input.lower().startswith("/concepts"):
                parts = user_input.split()
                limit = int(parts[1]) if len(parts) > 1 else 20
                result = await chat.concepts(limit)
                print(f"\n{result}")

            elif user_input.lower().startswith("/memories"):
                parts = user_input.split()
                limit = int(parts[1]) if len(parts) > 1 else 10
                result = await chat.memories(limit)
                print(f"\n{result}")

            elif user_input.lower().startswith("/episodes"):
                parts = user_input.split()
                limit = int(parts[1]) if len(parts) > 1 else 10
                result = await chat.episodes(limit)
                print(f"\n{result}")

            elif user_input.lower() == "/feedback +":
                result = await chat.feedback(positive=True)
                print(f"\n{result}")

            elif user_input.lower() == "/feedback -":
                result = await chat.feedback(positive=False)
                print(f"\n{result}")

            elif user_input.startswith("/"):
                print("Unknown command. Type /help for available commands.")

            else:
                # Regular chat
                print("\nEngram: ", end="", flush=True)
                result = await chat.chat(user_input)
                print(result)

    finally:
        await chat.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)
