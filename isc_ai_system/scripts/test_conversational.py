#!/usr/bin/env python3
"""
Unit tests for ISC Conversational Trainer
Tests all major components and functionality
"""

import unittest
from unittest.mock import Mock, MagicMock, patch, mock_open
import json
import os
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path
import numpy as np

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

# Import the module to test
from scripts.conversational import (
    ConversationalTask,
    ConversationalResult,
    ConversationalLMHead,
    ConversationalTrainer,
    ConversationalISC
)


class MockISCCore:
    """Mock ISCCore for testing"""
    def __init__(self):
        self.tokenizer = MagicMock()
        self.tokenizer.__len__ = Mock(return_value=50000)
        self.tokenizer.encode = Mock(return_value=[1, 2, 3, 4, 5])
        self.tokenizer.decode = Mock(return_value="generated text")
        
        self.network = MagicMock()
        self.network.hidden_dim = 768
        self.network.train = Mock()
        
        self.metrics = {
            "phi_value": 0.5,
            "emergence": 0.3,
            "integration": 0.4
        }
        
        self.knowledge_graph = MagicMock()
        self.knowledge_graph.graph = MagicMock()
        self.knowledge_graph.graph.nodes = Mock(return_value=['node1', 'node2'])
        self.knowledge_graph.graph.edges = Mock(return_value=[('node1', 'node2')])
        
        self.session_active = False
        self.current_session_id = ""
        
    def load_state(self, filepath):
        """Mock load state"""
        pass
        
    def save_state(self, filepath):
        """Mock save state"""
        pass
        
    def process_input(self, user_input, return_vector=False):
        """Mock process input with optional vector return"""
        response = f"Philosophical response to: {user_input}"
        if return_vector:
            # Return a mock vector of appropriate dimension
            vector = np.random.randn(self.network.hidden_dim).astype(np.float32)
            return response, vector
        return response


class TestConversationalDataClasses(unittest.TestCase):
    """Test data classes"""
    
    def test_conversational_task(self):
        """Test ConversationalTask dataclass"""
        task = ConversationalTask(
            exchange_id=1,
            topic="philosophy",
            prompt_type="dialogue",
            level=5,
            context="test context"
        )
        
        self.assertEqual(task.exchange_id, 1)
        self.assertEqual(task.topic, "philosophy")
        self.assertEqual(task.prompt_type, "dialogue")
        self.assertEqual(task.level, 5)
        self.assertEqual(task.context, "test context")
    
    def test_conversational_result(self):
        """Test ConversationalResult dataclass"""
        result = ConversationalResult(
            exchange_id=1,
            task_type="dialogue",
            content="test content",
            tokens_used={"prompt": 100, "completion": 50},
            error=None
        )
        
        self.assertEqual(result.exchange_id, 1)
        self.assertEqual(result.task_type, "dialogue")
        self.assertEqual(result.content, "test content")
        self.assertEqual(result.tokens_used["prompt"], 100)
        self.assertIsNone(result.error)


class TestConversationalLMHead(unittest.TestCase):
    """Test ConversationalLMHead neural network module"""
    
    def setUp(self):
        self.hidden_dim = 768
        self.vocab_size = 50000
        self.embedding_dim = 512
        self.lm_head = ConversationalLMHead(
            self.hidden_dim, 
            self.vocab_size, 
            self.embedding_dim
        )
    
    def test_initialization(self):
        """Test LM head initialization"""
        self.assertEqual(self.lm_head.hidden_dim, self.hidden_dim)
        self.assertEqual(self.lm_head.vocab_size, self.vocab_size)
        self.assertEqual(self.lm_head.embedding_dim, self.embedding_dim)
        
        # Check layer dimensions
        self.assertEqual(self.lm_head.concept_projection.in_features, self.hidden_dim)
        self.assertEqual(self.lm_head.concept_projection.out_features, self.embedding_dim)
        self.assertEqual(self.lm_head.output_projection.in_features, self.embedding_dim)
        self.assertEqual(self.lm_head.output_projection.out_features, self.vocab_size)
    
    def test_forward_without_context(self):
        """Test forward pass without context"""
        batch_size = 2
        concept_vector = torch.randn(batch_size, self.hidden_dim)
        
        output = self.lm_head(concept_vector)
        
        self.assertEqual(output.shape, (batch_size, self.vocab_size))
    
    def test_forward_with_context(self):
        """Test forward pass with context"""
        batch_size = 2
        seq_len = 5
        concept_vector = torch.randn(batch_size, self.hidden_dim)
        context_vectors = torch.randn(batch_size, seq_len, self.embedding_dim)
        
        output = self.lm_head(concept_vector, context_vectors)
        
        self.assertEqual(output.shape, (batch_size, self.vocab_size))


class TestConversationalTrainer(unittest.TestCase):
    """Test ConversationalTrainer class"""
    
    def setUp(self):
        """Set up test environment"""
        self.api_key = "test_api_key"
        
        # Mock OpenAI client
        self.mock_openai_patcher = patch('scripts.conversational.openai.OpenAI')
        self.mock_openai_class = self.mock_openai_patcher.start()
        self.mock_openai_client = MagicMock()
        self.mock_openai_class.return_value = self.mock_openai_client
        
        # Mock ISCCore
        self.mock_isc_patcher = patch('scripts.conversational.ISCCore')
        self.mock_isc_class = self.mock_isc_patcher.start()
        self.mock_isc_class.return_value = MockISCCore()
        
        # Create trainer
        self.trainer = ConversationalTrainer(self.api_key, max_workers=2)
    
    def tearDown(self):
        """Clean up patches"""
        self.mock_openai_patcher.stop()
        self.mock_isc_patcher.stop()
    
    def test_initialization(self):
        """Test trainer initialization"""
        self.assertEqual(self.trainer.max_workers, 2)
        self.assertEqual(self.trainer.learning_rate, 1e-4)
        self.assertEqual(self.trainer.batch_size, 4)
        self.assertEqual(self.trainer.auto_save_interval, 10)
        self.assertIsNotNone(self.trainer.openai_client)
        self.assertIsNotNone(self.trainer.isc)
    
    @patch('os.path.exists')
    @patch('torch.load')
    def test_load_model_with_existing_lm_head(self, mock_torch_load, mock_exists):
        """Test loading model with existing LM head"""
        mock_exists.return_value = True
        # Create proper state dict for LM head matching the trainer's expected dimensions
        # The trainer creates LM head with default embedding_dim=768
        lm_head = ConversationalLMHead(768, 50000, 768)  # hidden_dim=768, vocab_size=50000, embedding_dim=768
        mock_torch_load.return_value = lm_head.state_dict()
        
        result = self.trainer.load_model("test_model.pt")
        
        self.assertTrue(result)
        self.assertIsNotNone(self.trainer.lm_head)
        self.assertTrue(self.trainer.isc.session_active)
    
    @patch('os.path.exists')
    def test_load_model_without_lm_head(self, mock_exists):
        """Test loading model without existing LM head"""
        mock_exists.return_value = False
        
        result = self.trainer.load_model("test_model.pt")
        
        self.assertTrue(result)
        self.assertIsNotNone(self.trainer.lm_head)
        self.assertTrue(self.trainer.isc.session_active)
    
    def test_create_optimizer(self):
        """Test optimizer creation"""
        # Create LM head first
        self.trainer.lm_head = ConversationalLMHead(768, 50000, 512)
        
        self.trainer.create_optimizer()
        
        self.assertIsNotNone(self.trainer.optimizer)
        self.assertIsNotNone(self.trainer.scheduler)
    
    def test_generate_dialogue_examples(self):
        """Test dialogue example generation"""
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = """
Human: Hello, how are you?
AI: I'm doing well, thank you! How can I help you today?

Human: Tell me about consciousness.
AI: Consciousness is a fascinating topic that involves awareness and subjective experience."""
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        self.mock_openai_client.chat.completions.create.return_value = mock_response
        
        examples = self.trainer.generate_dialogue_examples("philosophy", 2)
        
        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0][0], "Hello, how are you?")
        self.assertEqual(examples[0][1], "I'm doing well, thank you! How can I help you today?")
    
    def test_generate_response_templates(self):
        """Test response template generation"""
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "greeting": "Hello! I've been thinking about {concept}. How can I help?",
            "question": "That's interesting about {concept}. {response}"
        })
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 30
        self.mock_openai_client.chat.completions.create.return_value = mock_response
        
        templates = self.trainer.generate_response_templates(2)
        
        self.assertIn("greeting", templates)
        self.assertIn("question", templates)
        self.assertIn("{concept}", templates["greeting"])
    
    def test_evaluate_response(self):
        """Test response evaluation"""
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "naturalness": 8,
            "relevance": 9,
            "engagement": 7,
            "conversational_flow": 8,
            "overall": 8,
            "feedback": "Good response"
        })
        mock_response.usage.prompt_tokens = 80
        mock_response.usage.completion_tokens = 20
        self.mock_openai_client.chat.completions.create.return_value = mock_response
        
        evaluation = self.trainer.evaluate_response(
            "What is consciousness?",
            "Consciousness is awareness.",
            "Consciousness is the state of being aware."
        )
        
        self.assertEqual(evaluation["naturalness"], 8)
        self.assertEqual(evaluation["overall"], 8)
        self.assertEqual(evaluation["feedback"], "Good response")
    
    def test_process_response(self):
        """Test process response through LM head"""
        # Create LM head
        self.trainer.lm_head = ConversationalLMHead(768, 50000, 512)
        
        # Create concept vector
        concept_vector = np.random.randn(768)
        
        # Test processing
        result = self.trainer.process_response(concept_vector)
        
        self.assertIsInstance(result, str)
        self.assertEqual(result, "generated text")
    
    def test_enhance_response(self):
        """Test response enhancement"""
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Enhanced philosophical response"
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        self.mock_openai_client.chat.completions.create.return_value = mock_response
        
        enhanced = self.trainer.enhance_response(
            "Original response",
            "User question"
        )
        
        self.assertEqual(enhanced, "Enhanced philosophical response")
    
    def test_train_batch(self):
        """Test batch training"""
        # Create LM head and optimizer
        self.trainer.lm_head = ConversationalLMHead(768, 50000, 512)
        # Remove verbose=True from scheduler creation
        self.trainer.optimizer = torch.optim.AdamW(self.trainer.lm_head.parameters(), lr=self.trainer.learning_rate)
        self.trainer.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.trainer.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Create examples
        examples = [
            ("Hello", "Hi there!"),
            ("How are you?", "I'm doing well!")
        ]
        
        # Train batch
        loss = self.trainer.train_batch(examples)
        
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0)
    
    @patch('os.makedirs')
    @patch('builtins.open', new_callable=mock_open)
    @patch('torch.save')
    def test_save_checkpoint(self, mock_torch_save, mock_file, mock_makedirs):
        """Test checkpoint saving"""
        # Set up trainer state
        self.trainer.lm_head = ConversationalLMHead(768, 50000, 512)
        self.trainer.training_history = [{"exchange": 1, "loss": 0.5}]
        self.trainer.session_metrics["conversational_score"] = [7.5]
        
        # Save checkpoint
        files = self.trainer.save_checkpoint(10)
        
        # Check that files were saved
        self.assertIn("json", files)
        self.assertIn("pt", files)
        self.assertIn("lm", files)
        mock_file.assert_called()
        mock_torch_save.assert_called()
    
    def test_calculate_cost(self):
        """Test cost calculation"""
        self.trainer.session_metrics["tokens_used"]["prompt"] = 1000
        self.trainer.session_metrics["tokens_used"]["completion"] = 500
        
        cost = self.trainer._calculate_cost()
        
        expected_cost = (1000 * 0.0005 / 1000) + (500 * 0.0015 / 1000)
        self.assertAlmostEqual(cost, expected_cost, places=4)
    
    def test_generate_progress_display(self):
        """Test progress display generation"""
        # Add some metrics
        self.trainer.session_metrics["conversational_score"] = [5.0, 6.0, 7.0]
        self.trainer.session_metrics["phi_progression"] = [0.4, 0.45, 0.5]
        self.trainer.session_metrics["tokens_used"] = {"prompt": 1000, "completion": 500}
        
        progress_text = self.trainer._generate_progress_display()
        
        self.assertIn("Conversational Score:", progress_text)
        self.assertIn("Φ Value:", progress_text)
        self.assertIn("Cost Tracking:", progress_text)


class TestConversationalISC(unittest.TestCase):
    """Test ConversationalISC class"""
    
    @patch('scripts.conversational.ISCCore')
    @patch('torch.load')
    @patch('os.path.exists')
    def setUp(self, mock_exists, mock_torch_load, mock_isc_class):
        """Set up test environment"""
        # Mock file existence
        mock_exists.return_value = True
        
        # Mock torch load
        mock_torch_load.return_value = {}
        
        # Mock ISCCore
        self.mock_isc = MockISCCore()
        mock_isc_class.return_value = self.mock_isc
        
        # Create ConversationalISC
        self.conv_isc = ConversationalISC("test_model.pt", "test_lm_head.pt")
    
    def test_initialization(self):
        """Test ConversationalISC initialization"""
        self.assertIsNotNone(self.conv_isc.isc)
        self.assertIsNotNone(self.conv_isc.lm_head)
        self.assertEqual(self.conv_isc.max_context_length, 5)
        self.assertIsInstance(self.conv_isc.response_templates, dict)
    
    def test_prepare_context_vectors(self):
        """Test context vector preparation"""
        # Add conversation history
        self.conv_isc.conversation_history = [
            {"user": "Hello", "response": "Hi", "timestamp": "2024-01-01"},
            {"user": "How are you?", "response": "Good", "timestamp": "2024-01-02"}
        ]
        
        context_vectors = self.conv_isc._prepare_context_vectors()
        
        self.assertIsNotNone(context_vectors)
        self.assertEqual(context_vectors.shape[0], 1)  # Batch dimension
        self.assertEqual(context_vectors.shape[1], 2)  # Sequence length
        self.assertEqual(context_vectors.shape[2], self.conv_isc.lm_head.embedding_dim)
    
    def test_classify_input(self):
        """Test input classification"""
        # Test greeting
        self.assertEqual(self.conv_isc._classify_input("Hello there!"), "greeting")
        
        # Test question
        self.assertEqual(self.conv_isc._classify_input("What is consciousness?"), "question")
        
        # Test philosophical
        self.assertEqual(self.conv_isc._classify_input("I think about philosophy"), "philosophical")
        
        # Test short input
        self.assertEqual(self.conv_isc._classify_input("Yes"), "continuation")
        
        # Test default
        self.assertEqual(self.conv_isc._classify_input("This is a longer statement"), "continuation")
    
    def test_chat(self):
        """Test chat functionality"""
        user_input = "What is consciousness?"
        
        conv_response, isc_response = self.conv_isc.chat(user_input)
        
        self.assertIsInstance(conv_response, str)
        self.assertIsInstance(isc_response, str)
        self.assertEqual(len(self.conv_isc.conversation_history), 1)
        self.assertEqual(self.conv_isc.conversation_history[0]["user"], user_input)
    
    def test_chat_history_limit(self):
        """Test that conversation history is limited"""
        # Mock the tokenizer decode method to return proper text
        self.mock_isc.tokenizer.decode = Mock(return_value=f"Generated response")
        
        # Add many conversations
        for i in range(20):
            self.conv_isc.chat(f"Question {i}")
        
        # Check that history is limited
        max_history = self.conv_isc.max_context_length * 2
        self.assertLessEqual(len(self.conv_isc.conversation_history), max_history)


class TestIntegration(unittest.TestCase):
    """Integration tests for the conversational system"""
    
    @patch('scripts.conversational.openai.OpenAI')
    @patch('scripts.conversational.ISCCore')
    def test_end_to_end_training_flow(self, mock_isc_class, mock_openai_class):
        """Test basic end-to-end training flow"""
        # Set up mocks
        mock_isc_class.return_value = MockISCCore()
        mock_openai_client = MagicMock()
        mock_openai_class.return_value = mock_openai_client
        
        # Mock OpenAI responses
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "greeting": "Hello {concept}!"
        })
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        # Create trainer
        trainer = ConversationalTrainer("test_key", max_workers=1)
        
        # Load model
        with patch('os.path.exists', return_value=False):
            trainer.load_model("test_model.pt")
        
        # Generate templates
        templates = trainer.generate_response_templates(1)
        
        self.assertIsInstance(templates, dict)
        self.assertGreater(len(templates), 0)


if __name__ == "__main__":
    unittest.main()