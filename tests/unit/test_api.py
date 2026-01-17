"""Unit tests for API endpoints."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from engram.api.main import create_app
from engram.api.routes import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    StatsResponse,
)


class TestChatCompletionModels:
    """Tests for chat completion request/response models."""

    def test_chat_message_user(self) -> None:
        """Test user message model."""
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_chat_message_assistant(self) -> None:
        """Test assistant message model."""
        msg = ChatMessage(role="assistant", content="Hi there")
        assert msg.role == "assistant"

    def test_chat_completion_request_defaults(self) -> None:
        """Test request with defaults."""
        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="test")]
        )
        assert req.model == "engram"
        assert req.temperature == 0.7
        assert req.stream is False
        assert req.top_k_memories == 10
        assert req.top_k_episodes == 3

    def test_chat_completion_request_custom(self) -> None:
        """Test request with custom values."""
        req = ChatCompletionRequest(
            model="custom",
            messages=[ChatMessage(role="user", content="test")],
            temperature=0.5,
            top_k_memories=20,
        )
        assert req.model == "custom"
        assert req.temperature == 0.5
        assert req.top_k_memories == 20

    def test_chat_completion_response(self) -> None:
        """Test response model."""
        resp = ChatCompletionResponse(
            id="test-123",
            created=1234567890,
            model="engram",
            choices=[],
            usage=ChatCompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            episode_id="ep-123",
            confidence=0.85,
        )
        assert resp.id == "test-123"
        assert resp.episode_id == "ep-123"
        assert resp.confidence == 0.85


class TestFeedbackModels:
    """Tests for feedback request/response models."""

    def test_feedback_request_positive(self) -> None:
        """Test positive feedback request."""
        req = FeedbackRequest(
            episode_id="ep-123",
            feedback="positive",
        )
        assert req.feedback == "positive"
        assert req.correction_text is None

    def test_feedback_request_correction(self) -> None:
        """Test correction feedback request."""
        req = FeedbackRequest(
            episode_id="ep-123",
            feedback="correction",
            correction_text="The correct answer is...",
        )
        assert req.feedback == "correction"
        assert req.correction_text is not None

    def test_feedback_response(self) -> None:
        """Test feedback response model."""
        resp = FeedbackResponse(
            success=True,
            feedback_type="positive",
            episode_id="ep-123",
            memories_strengthened=5,
            consolidation_triggered=False,
        )
        assert resp.success is True
        assert resp.memories_strengthened == 5


class TestAdminModels:
    """Tests for admin endpoint models."""

    def test_health_response(self) -> None:
        """Test health response model."""
        resp = HealthResponse(
            status="healthy",
            neo4j_connected=True,
        )
        assert resp.status == "healthy"
        assert resp.version == "0.1.0"

    def test_stats_response(self) -> None:
        """Test stats response model."""
        resp = StatsResponse(
            concepts_count=100,
            semantic_memories_count=500,
            episodic_memories_count=50,
            documents_count=20,
        )
        assert resp.concepts_count == 100


class TestChatCompletionEndpoint:
    """Tests for /v1/chat/completions endpoint."""

    @pytest.fixture
    def mock_db(self) -> MagicMock:
        """Create mock database."""
        db = MagicMock()
        db.connect = AsyncMock()
        db.close = AsyncMock()
        db.execute_query = AsyncMock(return_value=[{"n": 1}])
        return db

    @pytest.fixture
    def mock_reasoning_result(self) -> MagicMock:
        """Create mock reasoning result."""
        result = MagicMock()
        result.answer = "This is a test response"
        result.episode_id = "ep-test-123"
        result.confidence = 0.9
        result.synthesis = MagicMock()
        result.synthesis.concepts_activated = ["concept-1", "concept-2"]
        result.synthesis.memories_used = ["mem-1", "mem-2"]
        return result

    @pytest.mark.asyncio
    async def test_chat_completions_success(
        self, mock_db, mock_reasoning_result
    ) -> None:
        """Test successful chat completion."""
        with patch("engram.api.routes.ReasoningPipeline") as mock_pipeline_cls:
            mock_pipeline = MagicMock()
            mock_pipeline.reason = AsyncMock(return_value=mock_reasoning_result)
            mock_pipeline_cls.return_value = mock_pipeline

            app = create_app()
            app.state.db = mock_db

            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.post(
                    "/v1/chat/completions",
                    json={
                        "messages": [
                            {"role": "user", "content": "What is Docker?"}
                        ]
                    },
                )

                assert response.status_code == 200
                data = response.json()
                assert data["choices"][0]["message"]["content"] == "This is a test response"
                assert data["episode_id"] == "ep-test-123"
                assert data["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_chat_completions_no_user_message(self, mock_db) -> None:
        """Test chat completion with no user message."""
        app = create_app()
        app.state.db = mock_db

        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [
                        {"role": "system", "content": "You are a helper"}
                    ]
                },
            )

            assert response.status_code == 400
            assert "No user message" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_chat_completions_streaming_not_supported(self, mock_db) -> None:
        """Test that streaming is not yet supported."""
        app = create_app()
        app.state.db = mock_db

        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "test"}],
                    "stream": True,
                },
            )

            assert response.status_code == 501
            assert "Streaming" in response.json()["detail"]


class TestFeedbackEndpoint:
    """Tests for /v1/feedback endpoint."""

    @pytest.fixture
    def mock_db(self) -> MagicMock:
        """Create mock database."""
        db = MagicMock()
        db.connect = AsyncMock()
        db.close = AsyncMock()
        return db

    @pytest.fixture
    def mock_feedback_result(self) -> MagicMock:
        """Create mock feedback result."""
        result = MagicMock()
        result.success = True
        result.feedback_type = "positive"
        result.episode_id = "ep-123"
        result.memories_strengthened = 3
        result.concepts_strengthened = 2
        result.consolidation = MagicMock()
        result.consolidation.should_consolidate = False
        result.reflection = MagicMock()
        result.reflection.triggered = False
        result.memories_weakened = 0
        result.re_reasoning = None
        result.correction_memory = None
        return result

    @pytest.mark.asyncio
    async def test_feedback_positive(self, mock_db, mock_feedback_result) -> None:
        """Test positive feedback submission."""
        with patch("engram.api.routes.FeedbackHandler") as mock_handler_cls:
            mock_handler = MagicMock()
            mock_handler.handle_feedback = AsyncMock(return_value=mock_feedback_result)
            mock_handler_cls.return_value = mock_handler

            app = create_app()
            app.state.db = mock_db

            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.post(
                    "/v1/feedback",
                    json={
                        "episode_id": "ep-123",
                        "feedback": "positive",
                    },
                )

                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert data["memories_strengthened"] == 3
                assert data["consolidation_triggered"] is False

    @pytest.mark.asyncio
    async def test_feedback_negative(self, mock_db) -> None:
        """Test negative feedback submission."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.feedback_type = "negative"
        mock_result.episode_id = "ep-123"
        mock_result.memories_strengthened = 0
        mock_result.concepts_strengthened = 0
        mock_result.consolidation = None
        mock_result.reflection = None
        mock_result.memories_weakened = 2
        mock_result.re_reasoning = MagicMock()
        mock_result.re_reasoning.answer = "Alternative answer"
        mock_result.correction_memory = None

        with patch("engram.api.routes.FeedbackHandler") as mock_handler_cls:
            mock_handler = MagicMock()
            mock_handler.handle_feedback = AsyncMock(return_value=mock_result)
            mock_handler_cls.return_value = mock_handler

            app = create_app()
            app.state.db = mock_db

            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.post(
                    "/v1/feedback",
                    json={
                        "episode_id": "ep-123",
                        "feedback": "negative",
                    },
                )

                assert response.status_code == 200
                data = response.json()
                assert data["feedback_type"] == "negative"
                assert data["memories_weakened"] == 2
                assert data["alternative_answer"] == "Alternative answer"


class TestAdminEndpoints:
    """Tests for admin endpoints."""

    @pytest.fixture
    def mock_db(self) -> MagicMock:
        """Create mock database."""
        db = MagicMock()
        db.connect = AsyncMock()
        db.close = AsyncMock()
        db.execute_query = AsyncMock(return_value=[{"n": 1}])
        return db

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, mock_db) -> None:
        """Test health check when healthy."""
        with patch("engram.api.main.Neo4jClient") as mock_client_cls:
            mock_client_cls.return_value = mock_db

            app = create_app()

            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.get("/health")

                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "healthy"
                assert data["neo4j_connected"] is True

    @pytest.mark.asyncio
    async def test_health_check_degraded(self, mock_db) -> None:
        """Test health check when degraded."""
        mock_db.execute_query = AsyncMock(side_effect=Exception("Connection failed"))

        with patch("engram.api.main.Neo4jClient") as mock_client_cls:
            mock_client_cls.return_value = mock_db

            app = create_app()

            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.get("/health")

                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "degraded"
                assert data["neo4j_connected"] is False

    @pytest.mark.asyncio
    async def test_list_models(self, mock_db) -> None:
        """Test list models endpoint."""
        with patch("engram.api.main.Neo4jClient") as mock_client_cls:
            mock_client_cls.return_value = mock_db

            app = create_app()

            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.get("/v1/models")

                assert response.status_code == 200
                data = response.json()
                assert data["object"] == "list"
                assert len(data["data"]) == 1
                assert data["data"][0]["id"] == "engram"

    @pytest.mark.asyncio
    async def test_get_stats(self, mock_db) -> None:
        """Test get stats endpoint."""
        mock_db.execute_query = AsyncMock(return_value=[{
            "concepts": 100,
            "semantic": 500,
            "episodic": 50,
            "docs": 20,
        }])

        with patch("engram.api.main.Neo4jClient") as mock_client_cls:
            mock_client_cls.return_value = mock_db

            app = create_app()

            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.get("/admin/stats")

                assert response.status_code == 200
                data = response.json()
                assert data["concepts_count"] == 100
                assert data["semantic_memories_count"] == 500

    @pytest.mark.asyncio
    async def test_list_concepts(self, mock_db) -> None:
        """Test list concepts endpoint."""
        mock_db.execute_query = AsyncMock(return_value=[
            {"id": "c1", "name": "docker", "type": "tool", "activation_count": 10},
            {"id": "c2", "name": "kubernetes", "type": "tool", "activation_count": 5},
        ])

        with patch("engram.api.main.Neo4jClient") as mock_client_cls:
            mock_client_cls.return_value = mock_db

            app = create_app()

            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.get("/admin/concepts?limit=10")

                assert response.status_code == 200
                data = response.json()
                assert len(data) == 2
                assert data[0]["name"] == "docker"

    @pytest.mark.asyncio
    async def test_calibrate_memories(self, mock_db) -> None:
        """Test calibrate memories endpoint."""
        mock_db.execute_query = AsyncMock(return_value=[{"updated": 42}])

        with patch("engram.api.main.Neo4jClient") as mock_client_cls:
            mock_client_cls.return_value = mock_db

            app = create_app()

            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.post("/admin/calibrate?decay_factor=0.99")

                assert response.status_code == 200
                data = response.json()
                assert data["calibrated"] is True
                assert data["memories_updated"] == 42

    @pytest.mark.asyncio
    async def test_delete_episode(self, mock_db) -> None:
        """Test delete episode endpoint."""
        mock_db.execute_query = AsyncMock(return_value=[{"deleted": 1}])

        with patch("engram.api.main.Neo4jClient") as mock_client_cls:
            mock_client_cls.return_value = mock_db

            app = create_app()

            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.delete("/admin/episodes/ep-123")

                assert response.status_code == 200
                data = response.json()
                assert data["deleted"] is True
                assert data["episode_id"] == "ep-123"

    @pytest.mark.asyncio
    async def test_delete_episode_not_found(self, mock_db) -> None:
        """Test delete episode when not found."""
        mock_db.execute_query = AsyncMock(return_value=[{"deleted": 0}])

        with patch("engram.api.main.Neo4jClient") as mock_client_cls:
            mock_client_cls.return_value = mock_db

            app = create_app()

            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.delete("/admin/episodes/nonexistent")

                assert response.status_code == 404
