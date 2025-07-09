#!/usr/bin/env python3
"""
Unit tests for the conversational assessment tool
"""

import unittest
from unittest.mock import Mock, MagicMock, patch, mock_open
import json
import os
from pathlib import Path
from datetime import datetime
import numpy as np

# Import assessment components
from assess_conversational import (
    ConversationMetrics,
    ModelAssessment,
    ConversationalAssessor
)


class TestConversationMetrics(unittest.TestCase):
    """Test ConversationMetrics dataclass"""
    
    def test_overall_score_calculation(self):
        """Test overall score calculation with weights"""
        metrics = ConversationMetrics(
            exchange_id=1,
            human_input="Test input",
            model_response="Test response",
            response_time=0.5,
            coherence=8.0,
            relevance=9.0,
            naturalness=7.0,
            engagement=8.5,
            completeness=7.5,
            response_length=10,
            vocabulary_diversity=0.8,
            sentence_complexity=6.0,
            topic_consistency=8.0,
            context_utilization=7.0,
            philosophical_depth=9.0,
            conceptual_integration=8.0
        )
        
        # Calculate expected score based on weights
        expected = (
            0.20 * 8.0 +  # coherence
            0.20 * 9.0 +  # relevance
            0.15 * 7.0 +  # naturalness
            0.15 * 8.5 +  # engagement
            0.10 * 7.5 +  # completeness
            0.10 * 8.0 +  # topic_consistency
            0.10 * 9.0    # philosophical_depth
        )
        
        self.assertAlmostEqual(metrics.overall_score(), expected, places=2)


class TestConversationalAssessor(unittest.TestCase):
    """Test ConversationalAssessor class"""
    
    def setUp(self):
        """Set up test environment"""
        self.api_key = "test_api_key"
        
        # Mock OpenAI client
        with patch('assess_conversational.openai.OpenAI'):
            self.assessor = ConversationalAssessor(self.api_key)
            self.assessor.openai_client = MagicMock()
    
    def test_initialization(self):
        """Test assessor initialization"""
        self.assertEqual(self.assessor.api_key, self.api_key)
        self.assertIsNotNone(self.assessor.test_conversations)
        self.assertGreater(len(self.assessor.test_conversations), 10)
    
    def test_evaluate_response(self):
        """Test response evaluation"""
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "coherence": {"score": 8, "reason": "Well structured"},
            "relevance": {"score": 9, "reason": "On topic"},
            "naturalness": {"score": 7, "reason": "Mostly natural"},
            "engagement": {"score": 8, "reason": "Engaging"},
            "completeness": {"score": 8, "reason": "Complete"},
            "topic_consistency": {"score": 8, "reason": "Consistent"},
            "context_utilization": {"score": 7, "reason": "Good context use"},
            "philosophical_depth": {"score": 9, "reason": "Deep insights"},
            "conceptual_integration": {"score": 8, "reason": "Well integrated"},
            "vocabulary_diversity": {"score": 7, "reason": "Good variety"},
            "sentence_complexity": {"score": 6, "reason": "Appropriate"},
            "overall_impression": "Strong response"
        })
        self.assessor.openai_client.chat.completions.create.return_value = mock_response
        
        evaluation = self.assessor.evaluate_response(
            "What is consciousness?",
            "Consciousness is awareness of internal and external existence.",
            "philosophical",
            []
        )
        
        self.assertEqual(evaluation["coherence"]["score"], 8)
        self.assertEqual(evaluation["relevance"]["score"], 9)
        self.assertEqual(evaluation["overall_impression"], "Strong response")
    
    @patch('assess_conversational.ConversationalISC')
    def test_run_conversation_test(self, mock_conv_isc_class):
        """Test running a single conversation test"""
        # Mock the conversational model
        mock_model = MagicMock()
        mock_model.chat.return_value = ("This is a response", "Original response")
        
        # Mock evaluation
        self.assessor.evaluate_response = Mock(return_value={
            "coherence": {"score": 8, "reason": "Good"},
            "relevance": {"score": 9, "reason": "Good"},
            "naturalness": {"score": 7, "reason": "Good"},
            "engagement": {"score": 8, "reason": "Good"},
            "completeness": {"score": 8, "reason": "Good"},
            "topic_consistency": {"score": 8, "reason": "Good"},
            "context_utilization": {"score": 7, "reason": "Good"},
            "philosophical_depth": {"score": 9, "reason": "Good"},
            "conceptual_integration": {"score": 8, "reason": "Good"},
            "vocabulary_diversity": {"score": 7, "reason": "Good"},
            "sentence_complexity": {"score": 6, "reason": "Good"}
        })
        
        metrics, evaluation = self.assessor.run_conversation_test(
            mock_model,
            "Hello",
            "greeting",
            []
        )
        
        self.assertIsInstance(metrics, ConversationMetrics)
        self.assertEqual(metrics.human_input, "Hello")
        self.assertEqual(metrics.model_response, "This is a response")
        self.assertEqual(metrics.coherence, 8)
    
    def test_calculate_assessment(self):
        """Test assessment calculation from metrics"""
        # Create test metrics
        metrics_list = []
        for i in range(5):
            metrics = ConversationMetrics(
                exchange_id=i,
                human_input=f"Test {i}",
                model_response=f"Response {i}",
                response_time=0.5 + i * 0.1,
                coherence=7.0 + i * 0.2,
                relevance=8.0,
                naturalness=7.5,
                engagement=8.0,
                completeness=7.5,
                response_length=20,
                vocabulary_diversity=0.8,
                sentence_complexity=6.0,
                topic_consistency=8.0,
                context_utilization=7.0,
                philosophical_depth=8.5,
                conceptual_integration=8.0
            )
            metrics_list.append(metrics)
        
        evaluations = [{}] * 5  # Empty evaluations for test
        
        assessment = self.assessor._calculate_assessment(
            "test_model.pt",
            metrics_list,
            evaluations
        )
        
        self.assertIsInstance(assessment, ModelAssessment)
        self.assertEqual(assessment.total_exchanges, 5)
        self.assertAlmostEqual(assessment.avg_coherence, 7.4, places=1)
        self.assertEqual(assessment.avg_relevance, 8.0)
        self.assertGreater(assessment.coherence_trend, 0)  # Should be improving
    
    def test_trend_indicator(self):
        """Test trend indicator conversion"""
        self.assertEqual(self.assessor._trend_indicator(0.2), "📈 Improving")
        self.assertEqual(self.assessor._trend_indicator(-0.2), "📉 Declining")
        self.assertEqual(self.assessor._trend_indicator(0.05), "➡️ Stable")
    
    def test_rating(self):
        """Test score to rating conversion"""
        self.assertEqual(self.assessor._rating(8.5), "⭐ Excellent")
        self.assertEqual(self.assessor._rating(7.5), "✅ Good")
        self.assertEqual(self.assessor._rating(6.5), "⚠️ Fair")
        self.assertEqual(self.assessor._rating(5.5), "❌ Needs Work")
    
    def test_identify_weak_areas(self):
        """Test weak area identification"""
        metrics = ConversationMetrics(
            exchange_id=1,
            human_input="Test",
            model_response="Response",
            response_time=0.5,
            coherence=5.0,  # Weak
            relevance=8.0,
            naturalness=4.0,  # Weak
            engagement=8.0,
            completeness=7.0,
            response_length=10,
            vocabulary_diversity=0.8,
            sentence_complexity=6.0,
            topic_consistency=8.0,
            context_utilization=7.0,
            philosophical_depth=8.0,
            conceptual_integration=8.0
        )
        
        weak_areas = self.assessor._identify_weak_areas(metrics)
        self.assertIn("coherence", weak_areas)
        self.assertIn("naturalness", weak_areas)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_generate_plots(self, mock_close, mock_savefig):
        """Test plot generation"""
        # Create test metrics
        metrics_list = []
        for i in range(10):
            metrics = ConversationMetrics(
                exchange_id=i,
                human_input=f"Test {i}",
                model_response=f"Response {i}",
                response_time=0.5,
                coherence=7.0 + np.random.random(),
                relevance=8.0 + np.random.random(),
                naturalness=7.5 + np.random.random(),
                engagement=8.0,
                completeness=7.5,
                response_length=20 + i,
                vocabulary_diversity=0.8,
                sentence_complexity=6.0,
                topic_consistency=8.0,
                context_utilization=7.0,
                philosophical_depth=8.5,
                conceptual_integration=8.0
            )
            metrics_list.append(metrics)
        
        output_dir = Path("test_output")
        self.assessor.generate_plots(metrics_list, output_dir)
        
        # Check that plots were saved
        self.assertEqual(mock_savefig.call_count, 4)  # 4 different plots
        mock_close.assert_called()
    
    def test_generate_report(self):
        """Test report generation"""
        # Create test assessment
        assessment = ModelAssessment(
            model_path="test_model.pt",
            assessment_date=datetime.now().isoformat(),
            total_exchanges=10,
            avg_coherence=7.5,
            avg_relevance=8.0,
            avg_naturalness=7.0,
            avg_engagement=7.5,
            avg_completeness=7.0,
            avg_response_time=0.5,
            avg_response_length=25,
            coherence_trend=0.1,
            relevance_trend=0.0,
            naturalness_trend=-0.05,
            topic_diversity=0.8,
            context_consistency=7.5,
            philosophical_sophistication=8.0,
            strengths=["Relevance (8.0/10)", "Philosophy (8.0/10)"],
            weaknesses=["Naturalness (7.0/10)"],
            llm_summary="Test summary",
            llm_recommendations=["Improve naturalness", "Focus on engagement"]
        )
        
        # Create test metrics
        metrics_list = [
            ConversationMetrics(
                exchange_id=1,
                human_input="Test question",
                model_response="Test response that is quite long and detailed",
                response_time=0.5,
                coherence=8.0,
                relevance=9.0,
                naturalness=7.0,
                engagement=8.0,
                completeness=8.0,
                response_length=10,
                vocabulary_diversity=0.8,
                sentence_complexity=6.0,
                topic_consistency=8.0,
                context_utilization=7.0,
                philosophical_depth=9.0,
                conceptual_integration=8.0
            )
        ]
        
        output_dir = Path("test_output")
        report = self.assessor.generate_report(assessment, metrics_list, output_dir)
        
        # Check report content
        self.assertIn("# ISC Conversational Model Assessment Report", report)
        self.assertIn("test_model.pt", report)
        self.assertIn("Test summary", report)
        self.assertIn("Improve naturalness", report)
        self.assertIn("Performance Visualizations", report)
    
    def test_compare_models(self):
        """Test model comparison functionality"""
        assessments = [
            ModelAssessment(
                model_path="model1.pt",
                assessment_date=datetime.now().isoformat(),
                total_exchanges=10,
                avg_coherence=7.5,
                avg_relevance=8.0,
                avg_naturalness=7.0,
                avg_engagement=7.5,
                avg_completeness=7.0,
                avg_response_time=0.5,
                avg_response_length=25,
                coherence_trend=0.1,
                relevance_trend=0.0,
                naturalness_trend=-0.05,
                topic_diversity=0.8,
                context_consistency=7.5,
                philosophical_sophistication=8.0
            ),
            ModelAssessment(
                model_path="model2.pt",
                assessment_date=datetime.now().isoformat(),
                total_exchanges=10,
                avg_coherence=8.0,
                avg_relevance=8.5,
                avg_naturalness=7.5,
                avg_engagement=8.0,
                avg_completeness=7.5,
                avg_response_time=0.4,
                avg_response_length=30,
                coherence_trend=0.2,
                relevance_trend=0.1,
                naturalness_trend=0.1,
                topic_diversity=0.85,
                context_consistency=8.0,
                philosophical_sophistication=8.5
            )
        ]
        
        comparison = self.assessor.compare_models(assessments)
        
        self.assertIn("# Model Comparison Report", comparison)
        self.assertIn("model1.pt", comparison)
        self.assertIn("model2.pt", comparison)
        self.assertIn("Best Overall Model:", comparison)


if __name__ == "__main__":
    unittest.main()