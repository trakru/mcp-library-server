"""Ollama client for local LLM generation."""

import requests
import json
import logging
from typing import Dict, Any, List, Optional
import time

logger = logging.getLogger(__name__)

class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen2.5:7b"):
        """
        Initialize Ollama client.
        
        Args:
            base_url: Base URL for Ollama API
            model: Model name to use for generation
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.session = requests.Session()
        
        # Test connection and model availability
        self._verify_connection()
        self._verify_model()
        
        logger.info(f"Initialized Ollama client with model: {model}")
    
    def _verify_connection(self) -> None:
        """Verify connection to Ollama server."""
        try:
            response = self.session.get(f"{self.base_url}/api/version", timeout=5)
            if response.status_code == 200:
                version_info = response.json()
                logger.info(f"Connected to Ollama server version: {version_info.get('version', 'unknown')}")
            else:
                raise ConnectionError(f"Ollama server returned status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to Ollama server at {self.base_url}: {e}")
            raise ConnectionError(f"Cannot connect to Ollama server: {e}")
    
    def _verify_model(self) -> None:
        """Verify that the specified model is available."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model.get('name', '') for model in models]
                
                if self.model not in model_names:
                    logger.warning(f"Model {self.model} not found in available models: {model_names}")
                    logger.warning(f"You may need to run: ollama pull {self.model}")
                else:
                    logger.info(f"Model {self.model} is available")
            else:
                logger.warning(f"Could not verify model availability: {response.status_code}")
        except Exception as e:
            logger.warning(f"Error verifying model: {e}")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, 
                temperature: float = 0.7, max_tokens: int = 2048,
                stop_sequences: Optional[List[str]] = None) -> str:
        """
        Generate text using Ollama.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            stop_sequences: Stop sequences for generation
            
        Returns:
            Generated text
        """
        try:
            # Prepare the request payload
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            }
            
            # Add system prompt if provided
            if system_prompt:
                payload["system"] = system_prompt
            
            # Add stop sequences if provided
            if stop_sequences:
                payload["options"]["stop"] = stop_sequences
            
            logger.debug(f"Generating with prompt length: {len(prompt)} characters")
            
            # Make the request
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120  # 2 minute timeout for generation
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('response', '')
                
                # Log some stats
                total_duration = result.get('total_duration', 0) / 1e9  # Convert to seconds
                eval_count = result.get('eval_count', 0)
                
                logger.info(f"Generated {len(generated_text)} characters in {total_duration:.2f}s")
                if eval_count > 0:
                    logger.info(f"Generated {eval_count} tokens ({eval_count/total_duration:.1f} tokens/s)")
                
                return generated_text
            else:
                error_msg = f"Ollama API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
                
        except requests.exceptions.Timeout:
            logger.error("Ollama generation timed out")
            raise RuntimeError("Generation timed out")
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise
    
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7, 
             max_tokens: int = 2048) -> str:
        """
        Generate text using chat format.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response
        """
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            }
            
            logger.debug(f"Chat generation with {len(messages)} messages")
            
            response = self.session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                message = result.get('message', {})
                generated_text = message.get('content', '')
                
                logger.info(f"Generated chat response: {len(generated_text)} characters")
                return generated_text
            else:
                error_msg = f"Ollama chat API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
                
        except Exception as e:
            logger.error(f"Error during chat generation: {e}")
            raise
    
    def is_healthy(self) -> bool:
        """Check if Ollama server is healthy."""
        try:
            response = self.session.get(f"{self.base_url}/api/version", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def list_models(self) -> List[str]:
        """List available models."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model.get('name', '') for model in models]
            return []
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []


class DocumentGenerator:
    """High-level document generator using Ollama."""
    
    def __init__(self, ollama_client: OllamaClient):
        """Initialize with Ollama client."""
        self.client = ollama_client
        self.logger = logging.getLogger(__name__)
    
    def generate_section(self, topic: str, source_content: List[str], 
                        style: str = "technical", max_length: str = "standard") -> str:
        """
        Generate a documentation section based on source content.
        
        Args:
            topic: Topic to write about
            source_content: List of relevant source passages
            style: Writing style (technical, tutorial, overview)
            max_length: Length preference (brief, standard, comprehensive)
            
        Returns:
            Generated documentation section
        """
        # Prepare source context
        sources_text = "\n\n".join([f"Source {i+1}:\n{content}" for i, content in enumerate(source_content)])
        
        # Length guidelines
        length_guide = {
            "brief": "Write a concise summary in 1-2 paragraphs.",
            "standard": "Write a comprehensive section with 3-5 paragraphs including examples.",
            "comprehensive": "Write a detailed, thorough explanation with multiple sections, examples, and best practices."
        }
        
        # Style guidelines
        style_guide = {
            "technical": "Use precise technical language suitable for ML engineers and data scientists.",
            "tutorial": "Use clear, step-by-step explanations suitable for learning and implementation.",
            "overview": "Use accessible language suitable for stakeholders and decision-makers."
        }
        
        system_prompt = f"""You are an expert technical writer specializing in machine learning and data science documentation. 
        
Your task is to create high-quality documentation based on authoritative ML textbook sources.

Writing style: {style_guide.get(style, style_guide['technical'])}
Length requirement: {length_guide.get(max_length, length_guide['standard'])}

Guidelines:
- Base your content ONLY on the provided sources
- Include specific citations like (Source 1), (Source 2) etc.
- Maintain accuracy to the source material
- Structure content logically with clear sections
- Include practical insights and examples when available in sources
- Do not add information not present in the sources"""

        user_prompt = f"""Topic: {topic}

Source Material:
{sources_text}

Please write a documentation section on "{topic}" based solely on the provided source material. Include proper citations and maintain technical accuracy."""

        try:
            response = self.client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3,  # Lower temperature for factual content
                max_tokens=2048
            )
            
            self.logger.info(f"Generated section for topic: {topic}")
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating section for {topic}: {e}")
            raise
    
    def explain_concept(self, concept: str, source_content: List[str], 
                       include_examples: bool = True) -> str:
        """
        Generate an explanation of an ML concept.
        
        Args:
            concept: ML concept to explain
            source_content: Relevant source passages
            include_examples: Whether to include examples
            
        Returns:
            Generated explanation
        """
        sources_text = "\n\n".join([f"Source {i+1}:\n{content}" for i, content in enumerate(source_content)])
        
        example_instruction = "Include practical examples and use cases when available in the sources." if include_examples else "Focus on the conceptual explanation without detailed examples."
        
        system_prompt = f"""You are an expert ML educator. Explain machine learning concepts clearly and accurately based on authoritative textbook sources.

Guidelines:
- Explain the concept step by step
- Use clear, accessible language while maintaining technical accuracy
- Base explanations ONLY on the provided sources
- {example_instruction}
- Include citations (Source 1), (Source 2) etc.
- Structure the explanation logically"""

        user_prompt = f"""Concept to explain: {concept}

Source Material:
{sources_text}

Please provide a clear explanation of "{concept}" based on the source material provided."""

        try:
            response = self.client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.4,
                max_tokens=1500
            )
            
            self.logger.info(f"Generated explanation for concept: {concept}")
            return response
            
        except Exception as e:
            self.logger.error(f"Error explaining concept {concept}: {e}")
            raise
    
    def compare_approaches(self, approach1: str, approach2: str, 
                          source_content1: List[str], source_content2: List[str]) -> str:
        """
        Compare two ML approaches based on source content.
        
        Args:
            approach1: First approach name
            approach2: Second approach name
            source_content1: Sources for first approach
            source_content2: Sources for second approach
            
        Returns:
            Generated comparison
        """
        sources1_text = "\n\n".join([f"Source A{i+1}:\n{content}" for i, content in enumerate(source_content1)])
        sources2_text = "\n\n".join([f"Source B{i+1}:\n{content}" for i, content in enumerate(source_content2)])
        
        system_prompt = """You are an expert ML consultant. Compare ML approaches objectively based on authoritative sources.

Guidelines:
- Create a balanced comparison based ONLY on provided sources
- Structure the comparison with clear sections (Overview, Advantages, Disadvantages, Use Cases)
- Include proper citations
- Maintain objectivity and technical accuracy
- Highlight key differences and trade-offs"""

        user_prompt = f"""Compare these two approaches:

Approach 1: {approach1}
Sources for {approach1}:
{sources1_text}

Approach 2: {approach2}
Sources for {approach2}:
{sources2_text}

Please provide a comprehensive comparison of {approach1} vs {approach2} based on the source material."""

        try:
            response = self.client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=2048
            )
            
            self.logger.info(f"Generated comparison: {approach1} vs {approach2}")
            return response
            
        except Exception as e:
            self.logger.error(f"Error comparing {approach1} vs {approach2}: {e}")
            raise