"""
Tests for the Real Science modules of the Synthetic Cognition Platform.

Tests:
- Reasoning Execution (modus ponens, modus tollens, syllogisms)
- Cognitive Analysis (Toulmin model, Dual Process Theory)
- Uncertainty Quantification (Bayesian calibration)
- Comprehensive Benchmarks
"""

import pytest
import sys
sys.path.insert(0, 'src')


class TestReasoningExecution:
    """Tests for actual inference execution"""

    @pytest.fixture
    def inference_engine(self):
        from isc.reasoning_execution import RuleBasedInference
        return RuleBasedInference()

    def test_modus_ponens(self, inference_engine):
        """Test: If P→Q and P, then Q"""
        conditionals = [('it rains', 'the ground gets wet')]
        facts = ['raining', 'it rains']

        results = inference_engine.apply_modus_ponens(conditionals, facts)

        assert len(results) > 0
        derived_facts = [r[0] for r in results]
        assert 'the ground gets wet' in derived_facts

    def test_modus_tollens(self, inference_engine):
        """Test: If P→Q and ¬Q, then ¬P"""
        conditionals = [('it rains', 'the ground gets wet')]
        facts = ['not ground wet', 'the ground is not wet']

        results = inference_engine.apply_modus_tollens(conditionals, facts)

        assert len(results) > 0
        derived_facts = [r[0] for r in results]
        assert any('not' in f and 'rain' in f for f in derived_facts)

    def test_syllogism_positive(self, inference_engine):
        """Test: All A are B, All B are C → All A are C"""
        universals = [
            ('mammals', 'warm-blooded', True),  # All mammals are warm-blooded
            ('dogs', 'mammals', True),           # All dogs are mammals
        ]

        results = inference_engine.apply_syllogism(universals)

        assert len(results) > 0
        # Should derive: All dogs are warm-blooded
        found = any(
            subj == 'dogs' and 'warm' in pred and pos
            for subj, pred, pos, _ in results
        )
        assert found

    def test_syllogism_negative(self, inference_engine):
        """Test: No A are B, All C are A → No C are B"""
        universals = [
            ('reptiles', 'mammals', False),  # No reptiles are mammals
            ('snakes', 'reptiles', True),    # All snakes are reptiles
        ]

        results = inference_engine.apply_syllogism(universals)

        assert len(results) > 0
        # Should derive: No snakes are mammals
        found = any(
            'snake' in subj and 'mammal' in pred and not pos
            for subj, pred, pos, _ in results
        )
        assert found

    def test_parse_conditionals(self, inference_engine):
        """Test conditional parsing"""
        text = "If it rains, the ground gets wet."
        conditionals = inference_engine.parse_conditionals(text)

        assert len(conditionals) == 1
        antecedent, consequent = conditionals[0]
        assert 'rain' in antecedent
        assert 'wet' in consequent

    def test_parse_universals(self, inference_engine):
        """Test universal statement parsing"""
        text = "All mammals are warm-blooded. No reptiles are mammals."
        universals = inference_engine.parse_universals(text)

        assert len(universals) >= 2  # May extract additional patterns
        # Check positive universal
        assert any(pos for _, _, pos in universals)
        # Check negative universal
        assert any(not pos for _, _, pos in universals)

    def test_parse_facts_with_negation(self, inference_engine):
        """Test fact parsing including negations"""
        text = "The ground is not wet."
        facts = inference_engine.parse_facts(text)

        assert len(facts) > 0
        assert any('not' in f.lower() for f in facts)


class TestCognitiveAnalysis:
    """Tests for real cognitive analysis"""

    @pytest.fixture
    def analyzer(self):
        from isc.cognitive_analysis import RealCognitiveAnalyzer
        return RealCognitiveAnalyzer()

    @pytest.fixture
    def dual_process(self):
        from isc.cognitive_analysis import DualProcessAnalyzer
        return DualProcessAnalyzer()

    def test_system_2_detection(self, dual_process):
        """Test System 2 (analytical) detection"""
        text = "Therefore, based on logical analysis and the evidence, we can deduce that the hypothesis is correct."
        result = dual_process.analyze_dual_process(text)

        assert result['system_2_score'] > result['system_1_score']
        assert result['dominant_system'] == 2
        assert result['processing_style'] == 'analytical'

    def test_system_1_detection(self, dual_process):
        """Test System 1 (intuitive) detection"""
        text = "I feel like this is probably true. My gut says yes, it just seems right."
        result = dual_process.analyze_dual_process(text)

        # Should detect intuitive language
        assert result['system_1_score'] > 0

    def test_argument_extraction(self, analyzer):
        """Test Toulmin model argument extraction"""
        text = "All mammals are warm-blooded. Dogs are mammals. Therefore, dogs are warm-blooded."
        result = analyzer.analyze(text)

        components = result.get('argument_structure', {}).get('components', [])
        assert len(components) > 0

        # Check structure completeness
        structure = result.get('argument_structure', {})
        assert 'structure_completeness' in structure

    def test_coherence_analysis(self, analyzer):
        """Test semantic coherence analysis"""
        text = "Climate change is caused by greenhouse gases. Carbon dioxide is a greenhouse gas. Burning fossil fuels releases CO2."
        result = analyzer.analyze(text)

        coherence = result.get('coherence', {})
        assert 'overall' in coherence
        assert 0 <= coherence['overall'] <= 1


class TestUncertaintyQuantification:
    """Tests for Bayesian uncertainty estimation"""

    @pytest.fixture
    def estimator(self):
        from isc.uncertainty_quantification import ReasoningUncertaintyEstimator
        return ReasoningUncertaintyEstimator()

    def test_uncertainty_estimate_structure(self, estimator):
        """Test that uncertainty estimates have all required fields"""
        steps = [{'type': 'step', 'confidence': 0.9}]
        estimate = estimator.estimate_uncertainty(
            prediction='yes',
            raw_confidence=0.85,
            reasoning_steps=steps,
            problem_type='deductive'
        )

        assert hasattr(estimate, 'point_estimate')
        assert hasattr(estimate, 'confidence')
        assert hasattr(estimate, 'epistemic_uncertainty')
        assert hasattr(estimate, 'aleatoric_uncertainty')
        assert hasattr(estimate, 'credible_interval')

    def test_confidence_ordering(self, estimator):
        """Test that high raw confidence -> high final confidence"""
        steps = [{'type': 'step', 'confidence': 0.9}] * 3

        high = estimator.estimate_uncertainty('yes', 0.95, steps, 'deductive')
        low = estimator.estimate_uncertainty('yes', 0.50, steps, 'deductive')

        assert high.confidence > low.confidence

    def test_problem_type_adjustment(self, estimator):
        """Test that problem type affects confidence"""
        steps = [{'type': 'step', 'confidence': 0.9}]

        deductive = estimator.estimate_uncertainty('yes', 0.80, steps, 'deductive')
        abductive = estimator.estimate_uncertainty('yes', 0.80, steps, 'abductive')

        # Deductive should have higher confidence (it's more certain)
        assert deductive.confidence >= abductive.confidence

    def test_credible_interval_valid(self, estimator):
        """Test that credible intervals are valid"""
        steps = [{'type': 'step', 'confidence': 0.9}]
        estimate = estimator.estimate_uncertainty('yes', 0.85, steps, 'deductive')

        ci = estimate.credible_interval
        assert ci[0] <= ci[1]  # Lower bound <= upper bound
        assert ci[0] >= 0      # Non-negative
        assert ci[1] <= 1      # At most 1


class TestBayesianBeliefUpdater:
    """Tests for Bayesian belief updating"""

    @pytest.fixture
    def updater(self):
        from isc.uncertainty_quantification import BayesianBeliefUpdater
        return BayesianBeliefUpdater()

    def test_supporting_evidence_increases_belief(self, updater):
        """Test that supporting evidence increases belief"""
        prior = updater.get_belief('hypothesis')

        posterior = updater.update_belief(
            'hypothesis',
            'evidence_1',
            likelihood_ratio=3.0  # Evidence more likely under hypothesis
        )

        assert posterior > prior

    def test_opposing_evidence_decreases_belief(self, updater):
        """Test that opposing evidence decreases belief"""
        prior = updater.get_belief('hypothesis')

        posterior = updater.update_belief(
            'hypothesis',
            'evidence_2',
            likelihood_ratio=0.3  # Evidence less likely under hypothesis
        )

        assert posterior < prior


class TestComprehensiveBenchmarks:
    """Tests for the benchmark suite"""

    @pytest.fixture
    def benchmarks(self):
        from isc.comprehensive_benchmarks import ComprehensiveBenchmarks
        return ComprehensiveBenchmarks()

    def test_benchmark_count(self, benchmarks):
        """Test that we have 100+ problems"""
        all_problems = benchmarks.get_all_problems()
        assert len(all_problems) >= 100

    def test_all_reasoning_types_covered(self, benchmarks):
        """Test that all reasoning types have problems"""
        from isc.comprehensive_benchmarks import ReasoningType

        for rt in ReasoningType:
            problems = benchmarks.get_problems_by_type(rt)
            assert len(problems) > 0, f"No problems for {rt.value}"

    def test_problem_structure(self, benchmarks):
        """Test that problems have required fields"""
        problems = benchmarks.get_all_problems()

        for p in problems[:10]:  # Check first 10
            assert p.problem_id is not None
            assert p.premise is not None
            assert p.question is not None
            assert p.correct_answer is not None
            assert p.difficulty is not None


class TestIntegratedReasoning:
    """Integration tests for the full reasoning pipeline"""

    @pytest.fixture
    def core(self):
        from isc import NeuromorphicISCCore
        return NeuromorphicISCCore()

    def test_modus_ponens_integrated(self, core):
        """Test modus ponens through the full system"""
        result = core.execute_reasoning(
            premise='If it rains, the ground gets wet. It is raining.',
            question='Is the ground wet?'
        )

        assert result.get('answer', '').lower() == 'yes'
        assert result.get('success', False)

    def test_modus_tollens_integrated(self, core):
        """Test modus tollens through the full system"""
        result = core.execute_reasoning(
            premise='If it rains, the ground gets wet. The ground is not wet.',
            question='Is it raining?'
        )

        assert result.get('answer', '').lower() == 'no'

    def test_syllogism_integrated(self, core):
        """Test syllogistic reasoning through the full system"""
        result = core.execute_reasoning(
            premise='All mammals are warm-blooded. All dogs are mammals.',
            question='Are dogs warm-blooded?'
        )

        assert result.get('answer', '').lower() == 'yes'

    def test_negative_syllogism_integrated(self, core):
        """Test negative syllogism through the full system"""
        result = core.execute_reasoning(
            premise='No reptiles are mammals. All snakes are reptiles.',
            question='Are snakes mammals?'
        )

        assert result.get('answer', '').lower() == 'no'

    def test_chained_reasoning(self, core):
        """Test chained inference"""
        result = core.execute_reasoning(
            premise='All A are B. All B are C. All C are D.',
            question='Are A D?'
        )

        assert result.get('answer', '').lower() == 'yes'

    def test_abductive_best_explanation(self, core):
        """Test abductive reasoning (inference to best explanation)"""
        result = core.execute_reasoning(
            premise='The grass is wet. It is morning. There are no sprinklers running.',
            question='What is the most likely explanation?'
        )

        assert result.get('reasoning_type') == 'abductive'
        assert 'dew' in result.get('answer', '').lower()

    def test_abductive_diagnosis(self, core):
        """Test abductive reasoning for diagnosis"""
        result = core.execute_reasoning(
            premise='Patient has fever, cough, and body aches. It is flu season.',
            question='What is the most likely diagnosis?'
        )

        assert result.get('reasoning_type') == 'abductive'
        assert 'flu' in result.get('answer', '').lower()

    def test_abductive_debugging(self, core):
        """Test abductive reasoning for technical debugging"""
        result = core.execute_reasoning(
            premise="The car won't start. The lights don't turn on. The radio doesn't work.",
            question='What is the most likely problem?'
        )

        assert result.get('reasoning_type') == 'abductive'
        assert 'battery' in result.get('answer', '').lower()

    def test_analogical_habitat(self, core):
        """Test analogical reasoning for habitat relations"""
        result = core.execute_reasoning(
            premise='Bird is to sky as fish is to ?',
            question='Complete the analogy.'
        )

        assert result.get('reasoning_type') == 'analogical'
        assert 'water' in result.get('answer', '').lower()

    def test_analogical_tool_user(self, core):
        """Test analogical reasoning for tool-user relations"""
        result = core.execute_reasoning(
            premise='Pen is to writer as brush is to ?',
            question='Complete the analogy.'
        )

        assert result.get('reasoning_type') == 'analogical'
        assert 'painter' in result.get('answer', '').lower()

    def test_causal_confounding(self, core):
        """Test causal reasoning for spurious correlation"""
        result = core.execute_reasoning(
            premise='Ice cream sales increase in summer. Drowning incidents increase in summer.',
            question='Does ice cream cause drowning?'
        )

        assert result.get('reasoning_type') == 'causal'
        assert result.get('answer', '').lower() == 'no'

    def test_causal_rct(self, core):
        """Test causal reasoning for randomized trials"""
        result = core.execute_reasoning(
            premise='Randomized trial: Group A got the drug, Group B got placebo. Group A improved more.',
            question='Did the drug cause the improvement?'
        )

        assert result.get('reasoning_type') == 'causal'
        assert result.get('answer', '').lower() == 'yes'

    def test_uncertainty_integrated(self, core):
        """Test uncertainty estimation through the full system"""
        estimate = core.estimate_uncertainty(
            prediction='yes',
            raw_confidence=0.85,
            reasoning_steps=[{'type': 'derive', 'confidence': 0.9}],
            problem_type='deductive'
        )

        assert 'confidence' in estimate
        assert 'credible_interval' in estimate
        assert 0 < estimate['confidence'] < 1

    def test_cognitive_analysis_integrated(self, core):
        """Test cognitive analysis through the full system"""
        analysis = core.analyze_reasoning(
            'Therefore, based on the data, we conclude that the hypothesis is supported.'
        )

        assert 'dual_process' in analysis
        assert 'argument_structure' in analysis
