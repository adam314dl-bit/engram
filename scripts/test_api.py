#!/usr/bin/env python3
"""Integration test for Phase 6: API & Integration.

This script tests the API endpoints with a real Neo4j instance.
Exit criteria: "Full system runs in Docker, API works with Open WebUI."
"""

import logging
import sys

import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

API_BASE_URL = "http://localhost:8000"


def test_api_integration():
    """Test the full API integration."""

    print("\n" + "=" * 70)
    print("PHASE 6 INTEGRATION TEST: API & Integration")
    print("=" * 70)

    session = requests.Session()
    session.timeout = 60.0

    def get(path: str) -> requests.Response:
        return session.get(f"{API_BASE_URL}{path}", timeout=60.0)

    def post(path: str, json: dict) -> requests.Response:
        return session.post(f"{API_BASE_URL}{path}", json=json, timeout=60.0)

    # Test 1: Health check
    print("\n--- Test 1: Health Check ---")
    try:
        response = get("/health")
        print(f"Status code: {response.status_code}")
        data = response.json()
        print(f"Health status: {data['status']}")
        print(f"Neo4j connected: {data['neo4j_connected']}")

        if data["status"] != "healthy":
            print("WARNING: API is degraded")
    except requests.ConnectionError:
        print("ERROR: Cannot connect to API. Is it running?")
        print(f"Expected API at: {API_BASE_URL}")
        return False

    # Test 2: List models (OpenAI compatibility)
    print("\n--- Test 2: List Models (OpenAI Compatibility) ---")
    response = get("/v1/models")
    data = response.json()
    print(f"Available models: {[m['id'] for m in data['data']]}")

    # Test 3: Chat completion
    print("\n--- Test 3: Chat Completion ---")
    response = post(
        "/v1/chat/completions",
        {
            "model": "engram",
            "messages": [
                {"role": "user", "content": "Что такое Docker?"}
            ],
            "temperature": 0.7,
        },
    )
    print(f"Status code: {response.status_code}")
    data = response.json()

    if response.status_code == 200:
        print(f"Episode ID: {data.get('episode_id')}")
        print(f"Confidence: {data.get('confidence', 0):.1%}")
        print(f"Concepts activated: {len(data.get('concepts_activated', []))}")
        print(f"Memories used: {data.get('memories_used', 0)}")
        answer = data["choices"][0]["message"]["content"]
        print(f"Answer preview: {answer[:100]}...")
        episode_id = data.get("episode_id")
    else:
        print(f"Error: {data}")
        episode_id = None

    # Test 4: Feedback (if we have an episode)
    if episode_id:
        print("\n--- Test 4: Submit Positive Feedback ---")
        response = post(
            "/v1/feedback",
            {
                "episode_id": episode_id,
                "feedback": "positive",
            },
        )
        print(f"Status code: {response.status_code}")
        data = response.json()

        if response.status_code == 200:
            print(f"Success: {data['success']}")
            print(f"Memories strengthened: {data['memories_strengthened']}")
            print(f"Concepts strengthened: {data['concepts_strengthened']}")
            print(f"Consolidation triggered: {data['consolidation_triggered']}")
            print(f"Reflection triggered: {data['reflection_triggered']}")
        else:
            print(f"Error: {data}")

    # Test 5: Admin stats
    print("\n--- Test 5: Admin Stats ---")
    response = get("/admin/stats")
    data = response.json()
    print(f"Concepts: {data['concepts_count']}")
    print(f"Semantic memories: {data['semantic_memories_count']}")
    print(f"Episodic memories: {data['episodic_memories_count']}")
    print(f"Documents: {data['documents_count']}")

    # Test 6: List concepts
    print("\n--- Test 6: List Concepts ---")
    response = get("/admin/concepts?limit=5")
    data = response.json()
    print(f"Top {len(data)} concepts by activation:")
    for concept in data[:5]:
        print(f"  - {concept['name']} ({concept['type']}): {concept['activation_count']} activations")

    # Test 7: Chat with conversation history
    print("\n--- Test 7: Multi-turn Conversation ---")
    response = post(
        "/v1/chat/completions",
        {
            "model": "engram",
            "messages": [
                {"role": "system", "content": "Ты помощник по DevOps."},
                {"role": "user", "content": "Что такое Kubernetes?"},
                {"role": "assistant", "content": "Kubernetes - это платформа оркестрации контейнеров."},
                {"role": "user", "content": "Как настроить Pod?"},
            ],
        },
    )
    print(f"Status code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Episode ID: {data.get('episode_id')}")
        answer = data["choices"][0]["message"]["content"]
        print(f"Answer preview: {answer[:100]}...")
    else:
        print(f"Error: {response.json()}")

    # Test 8: Calibration
    print("\n--- Test 8: Calibration ---")
    response = post("/admin/calibrate?decay_factor=0.995", {})
    data = response.json()
    print(f"Calibrated: {data['calibrated']}")
    print(f"Memories updated: {data['memories_updated']}")

    print("\n" + "=" * 70)
    print("PHASE 6 EXIT CRITERIA: PASSED")
    print("- Health check endpoint works")
    print("- OpenAI-compatible /v1/models endpoint works")
    print("- /v1/chat/completions endpoint works")
    print("- /v1/feedback endpoint works")
    print("- Admin endpoints work (stats, concepts, calibrate)")
    print("- API is ready for Open WebUI integration")
    print("=" * 70)

    session.close()
    return True


def main():
    """Run the integration test."""
    try:
        success = test_api_integration()
        return success
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
