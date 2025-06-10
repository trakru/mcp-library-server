import pytest
import requests
from unittest.mock import Mock, patch, MagicMock
import json

from src.generation.ollama_client import OllamaClient, DocumentGenerator


class TestOllamaClient:
    """Test suite for the OllamaClient class."""
    
    @pytest.fixture
    def mock_session(self):
        """Create a mock requests session."""
        session = Mock(spec=requests.Session)
        
        # Mock successful version response
        version_response = Mock()
        version_response.status_code = 200
        version_response.json.return_value = {"version": "0.1.0"}
        
        # Mock successful models response
        models_response = Mock()
        models_response.status_code = 200
        models_response.json.return_value = {
            "models": [
                {"name": "qwen2.5:7b"},
                {"name": "llama2:7b"}
            ]
        }
        
        # Configure session responses
        def mock_get(url, **kwargs):
            if "version" in url:
                return version_response
            elif "tags" in url:
                return models_response
            return Mock(status_code=404)
        
        session.get.side_effect = mock_get
        return session
    
    @pytest.fixture
    def ollama_client(self, mock_session):
        """Create OllamaClient with mocked session."""
        with patch('src.generation.ollama_client.requests.Session', return_value=mock_session):
            client = OllamaClient(base_url="http://localhost:11434", model="qwen2.5:7b")
            return client
    
    def test_initialization_success(self, mock_session):
        """Test successful client initialization."""
        with patch('src.generation.ollama_client.requests.Session', return_value=mock_session):
            client = OllamaClient()
            
            assert client.base_url == "http://localhost:11434"
            assert client.model == "qwen2.5:7b"
            assert client.session == mock_session
    
    def test_initialization_connection_error(self):
        """Test initialization with connection error."""
        mock_session = Mock()
        mock_session.get.side_effect = requests.exceptions.ConnectionError("Connection failed")
        
        with patch('src.generation.ollama_client.requests.Session', return_value=mock_session):
            with pytest.raises(ConnectionError) as exc_info:
                OllamaClient()
            assert "Cannot connect to Ollama server" in str(exc_info.value)
    
    def test_initialization_server_error(self):
        """Test initialization with server error."""
        mock_session = Mock()
        error_response = Mock()
        error_response.status_code = 500
        mock_session.get.return_value = error_response
        
        with patch('src.generation.ollama_client.requests.Session', return_value=mock_session):
            with pytest.raises(ConnectionError) as exc_info:
                OllamaClient()
            assert "status code: 500" in str(exc_info.value)
    
    def test_verify_model_not_available(self, mock_session):
        """Test model verification when model is not available."""
        # Modify models response to not include our model
        models_response = Mock()
        models_response.status_code = 200
        models_response.json.return_value = {
            "models": [{"name": "llama2:7b"}]  # Different model
        }
        
        def mock_get(url, **kwargs):
            if "version" in url:
                version_response = Mock()
                version_response.status_code = 200
                version_response.json.return_value = {"version": "0.1.0"}
                return version_response
            elif "tags" in url:
                return models_response
            return Mock(status_code=404)
        
        mock_session.get.side_effect = mock_get
        
        with patch('src.generation.ollama_client.requests.Session', return_value=mock_session):
            # Should not raise an error, just log a warning
            client = OllamaClient(model="qwen2.5:7b")
            assert client.model == "qwen2.5:7b"
    
    def test_generate_success(self, ollama_client):
        """Test successful text generation."""
        # Mock successful generation response
        gen_response = Mock()
        gen_response.status_code = 200
        gen_response.json.return_value = {
            "response": "Generated text response",
            "total_duration": 2500000000,  # 2.5 seconds in nanoseconds
            "eval_count": 50
        }
        
        ollama_client.session.post.return_value = gen_response
        
        result = ollama_client.generate(
            prompt="Test prompt",
            system_prompt="Test system",
            temperature=0.8,
            max_tokens=1000
        )
        
        assert result == "Generated text response"
        
        # Verify request was made correctly
        ollama_client.session.post.assert_called_once()
        call_args = ollama_client.session.post.call_args
        
        assert "api/generate" in call_args[0][0]
        payload = call_args[1]['json']
        assert payload['model'] == "qwen2.5:7b"
        assert payload['prompt'] == "Test prompt"
        assert payload['system'] == "Test system"
        assert payload['options']['temperature'] == 0.8
        assert payload['options']['num_predict'] == 1000
    
    def test_generate_with_stop_sequences(self, ollama_client):
        """Test generation with stop sequences."""
        gen_response = Mock()
        gen_response.status_code = 200
        gen_response.json.return_value = {"response": "Generated text"}
        
        ollama_client.session.post.return_value = gen_response
        
        ollama_client.generate(
            prompt="Test prompt",
            stop_sequences=["END", "STOP"]
        )
        
        payload = ollama_client.session.post.call_args[1]['json']
        assert payload['options']['stop'] == ["END", "STOP"]
    
    def test_generate_api_error(self, ollama_client):
        """Test generation with API error."""
        error_response = Mock()
        error_response.status_code = 400
        error_response.text = "Bad request"
        
        ollama_client.session.post.return_value = error_response
        
        with pytest.raises(RuntimeError) as exc_info:
            ollama_client.generate("Test prompt")
        assert "Ollama API error: 400" in str(exc_info.value)
    
    def test_generate_timeout(self, ollama_client):
        """Test generation timeout."""
        ollama_client.session.post.side_effect = requests.exceptions.Timeout()
        
        with pytest.raises(RuntimeError) as exc_info:
            ollama_client.generate("Test prompt")
        assert "Generation timed out" in str(exc_info.value)
    
    def test_chat_success(self, ollama_client):
        """Test successful chat generation."""
        chat_response = Mock()
        chat_response.status_code = 200
        chat_response.json.return_value = {
            "message": {"content": "Chat response content"}
        }
        
        ollama_client.session.post.return_value = chat_response
        
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        result = ollama_client.chat(messages, temperature=0.5, max_tokens=500)
        
        assert result == "Chat response content"
        
        # Verify request
        call_args = ollama_client.session.post.call_args
        assert "api/chat" in call_args[0][0]
        payload = call_args[1]['json']
        assert payload['messages'] == messages
        assert payload['options']['temperature'] == 0.5
    
    def test_chat_api_error(self, ollama_client):
        """Test chat with API error."""
        error_response = Mock()
        error_response.status_code = 500
        error_response.text = "Server error"
        
        ollama_client.session.post.return_value = error_response
        
        with pytest.raises(RuntimeError) as exc_info:
            ollama_client.chat([{"role": "user", "content": "test"}])
        assert "Ollama chat API error: 500" in str(exc_info.value)
    
    def test_is_healthy_true(self, ollama_client):
        """Test health check when server is healthy."""
        health_response = Mock()
        health_response.status_code = 200
        
        ollama_client.session.get.return_value = health_response
        
        assert ollama_client.is_healthy() is True
    
    def test_is_healthy_false(self, ollama_client):
        """Test health check when server is unhealthy."""
        ollama_client.session.get.side_effect = requests.exceptions.ConnectionError()
        
        assert ollama_client.is_healthy() is False
    
    def test_list_models_success(self, ollama_client):
        """Test listing models successfully."""
        models_response = Mock()
        models_response.status_code = 200
        models_response.json.return_value = {
            "models": [
                {"name": "qwen2.5:7b"},
                {"name": "llama2:7b"},
                {"name": "mistral:7b"}
            ]
        }
        
        # Need to override the mock_get method from fixture
        def mock_get(url, **kwargs):
            if "tags" in url:
                return models_response
            # Return the default fixture behavior for other URLs
            version_response = Mock()
            version_response.status_code = 200
            version_response.json.return_value = {"version": "0.1.0"}
            return version_response
        
        ollama_client.session.get.side_effect = mock_get
        
        models = ollama_client.list_models()
        
        assert models == ["qwen2.5:7b", "llama2:7b", "mistral:7b"]
    
    def test_list_models_error(self, ollama_client):
        """Test listing models with error."""
        ollama_client.session.get.side_effect = requests.exceptions.RequestException()
        
        models = ollama_client.list_models()
        
        assert models == []


class TestDocumentGenerator:
    """Test suite for the DocumentGenerator class."""
    
    @pytest.fixture
    def mock_ollama_client(self):
        """Create a mock OllamaClient."""
        client = Mock(spec=OllamaClient)
        client.generate.return_value = "Generated documentation content"
        return client
    
    @pytest.fixture
    def doc_generator(self, mock_ollama_client):
        """Create DocumentGenerator with mock client."""
        return DocumentGenerator(mock_ollama_client)
    
    def test_generate_section(self, doc_generator, mock_ollama_client):
        """Test generating a documentation section."""
        topic = "Neural Networks"
        source_content = [
            "Neural networks are computational models...",
            "Backpropagation is the key training algorithm..."
        ]
        
        result = doc_generator.generate_section(
            topic=topic,
            source_content=source_content,
            style="technical",
            max_length="standard"
        )
        
        assert result == "Generated documentation content"
        
        # Verify the call to generate
        mock_ollama_client.generate.assert_called_once()
        call_args = mock_ollama_client.generate.call_args
        
        assert call_args[1]['temperature'] == 0.3
        assert call_args[1]['max_tokens'] == 2048
        assert topic in call_args[1]['prompt']
        assert "Source 1:" in call_args[1]['prompt']
        assert "Source 2:" in call_args[1]['prompt']
    
    def test_generate_section_different_styles(self, doc_generator, mock_ollama_client):
        """Test generating sections with different styles."""
        topic = "Machine Learning"
        source_content = ["Test content"]
        
        # Test tutorial style
        doc_generator.generate_section(topic, source_content, style="tutorial")
        system_prompt = mock_ollama_client.generate.call_args[1]['system_prompt']
        assert "step-by-step" in system_prompt
        
        # Test overview style
        doc_generator.generate_section(topic, source_content, style="overview")
        system_prompt = mock_ollama_client.generate.call_args[1]['system_prompt']
        assert "accessible language" in system_prompt
    
    def test_generate_section_different_lengths(self, doc_generator, mock_ollama_client):
        """Test generating sections with different lengths."""
        topic = "Deep Learning"
        source_content = ["Test content"]
        
        # Test brief length
        doc_generator.generate_section(topic, source_content, max_length="brief")
        system_prompt = mock_ollama_client.generate.call_args[1]['system_prompt']
        assert "1-2 paragraphs" in system_prompt
        
        # Test comprehensive length
        doc_generator.generate_section(topic, source_content, max_length="comprehensive")
        system_prompt = mock_ollama_client.generate.call_args[1]['system_prompt']
        assert "detailed, thorough" in system_prompt
    
    def test_explain_concept(self, doc_generator, mock_ollama_client):
        """Test explaining a concept."""
        concept = "Gradient Descent"
        source_content = ["Gradient descent is an optimization algorithm..."]
        
        result = doc_generator.explain_concept(
            concept=concept,
            source_content=source_content,
            include_examples=True
        )
        
        assert result == "Generated documentation content"
        
        call_args = mock_ollama_client.generate.call_args
        assert call_args[1]['temperature'] == 0.4
        assert call_args[1]['max_tokens'] == 1500
        assert concept in call_args[1]['prompt']
        assert "examples" in call_args[1]['system_prompt']
    
    def test_explain_concept_without_examples(self, doc_generator, mock_ollama_client):
        """Test explaining a concept without examples."""
        concept = "Overfitting"
        source_content = ["Test content"]
        
        doc_generator.explain_concept(concept, source_content, include_examples=False)
        
        system_prompt = mock_ollama_client.generate.call_args[1]['system_prompt']
        assert "without detailed examples" in system_prompt
    
    def test_compare_approaches(self, doc_generator, mock_ollama_client):
        """Test comparing two approaches."""
        approach1 = "Random Forest"
        approach2 = "Neural Networks"
        source_content1 = ["Random Forest is an ensemble method..."]
        source_content2 = ["Neural networks use layered architectures..."]
        
        result = doc_generator.compare_approaches(
            approach1=approach1,
            approach2=approach2,
            source_content1=source_content1,
            source_content2=source_content2
        )
        
        assert result == "Generated documentation content"
        
        call_args = mock_ollama_client.generate.call_args
        prompt = call_args[1]['prompt']
        
        assert approach1 in prompt
        assert approach2 in prompt
        assert "Source A1:" in prompt
        assert "Source B1:" in prompt
        assert call_args[1]['temperature'] == 0.3
    
    def test_generation_error_handling(self, doc_generator, mock_ollama_client):
        """Test error handling in document generation."""
        mock_ollama_client.generate.side_effect = Exception("Generation failed")
        
        with pytest.raises(Exception) as exc_info:
            doc_generator.generate_section("Test", ["content"])
        assert "Generation failed" in str(exc_info.value)
        
        with pytest.raises(Exception):
            doc_generator.explain_concept("Test", ["content"])
        
        with pytest.raises(Exception):
            doc_generator.compare_approaches("A", "B", ["content1"], ["content2"])