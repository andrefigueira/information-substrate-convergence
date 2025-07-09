# ISC Conversational Model Assessment Tool

A comprehensive evaluation and reporting system for ISC conversational models that produces detailed assessments with metrics, visualizations, and LLM-generated insights.

## Features

- **Comprehensive Testing**: Runs 15+ different conversation scenarios
- **Multi-dimensional Evaluation**: Assesses coherence, relevance, naturalness, engagement, and philosophical depth
- **Automated Visualizations**: Generates 4 types of performance plots
- **LLM-Based Analysis**: Uses GPT-3.5 to provide qualitative assessment
- **Detailed Reports**: Creates markdown reports with metrics, trends, and recommendations
- **Model Comparison**: Can compare multiple models side-by-side

## Setup

1. **Install Dependencies**:
```bash
pip install torch openai rich matplotlib pandas numpy
# Optional for better plots:
pip install seaborn
```

2. **Set OpenAI API Key**:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Running an Assessment

```bash
python assess_conversational.py
```

The tool will:
1. Search for trained conversational models
2. Display available models for selection
3. Run comprehensive tests (takes 2-3 minutes)
4. Generate visualizations and report
5. Save results in `conversational_reports/{timestamp}/`

### Output Structure

Each assessment creates a timestamped directory containing:
```
conversational_reports/20240109_143022/
├── assessment_report.md      # Main markdown report
├── assessment_data.json      # Raw assessment data
├── model_info.txt           # Model metadata
├── overall_scores.png       # Quality trend plot
├── performance_radar.png    # Multi-dimensional performance
├── response_analysis.png    # Response characteristics
└── metric_trends.png       # Individual metric trends
```

## Assessment Metrics

### Core Metrics (0-10 scale)
- **Coherence**: Internal consistency and structure
- **Relevance**: How well responses address inputs
- **Naturalness**: Human-like conversation quality
- **Engagement**: Ability to move conversation forward
- **Completeness**: Thoroughness without verbosity

### Advanced Metrics
- **Philosophical Depth**: ISC-specific philosophical insights
- **Context Utilization**: Use of conversation history
- **Topic Consistency**: Staying on topic or smooth transitions
- **Conceptual Integration**: Meaningful concept connections

### Performance Metrics
- **Response Time**: Speed of generation
- **Response Length**: Word count analysis
- **Vocabulary Diversity**: Lexical variety
- **Sentence Complexity**: Structural sophistication

## Test Scenarios

The assessment includes diverse conversation types:
- Basic greetings and small talk
- Philosophical questions
- Educational queries
- Emotional responses
- Complex technical discussions
- Context-dependent continuations
- Edge cases (empty input, nonsense)

## Report Sections

1. **Executive Summary**: LLM-generated overview
2. **Performance Metrics**: Aggregate scores with trends
3. **Strengths & Weaknesses**: Top performing and improvement areas
4. **Detailed Analysis**: Best/worst exchanges with examples
5. **Recommendations**: Actionable improvement suggestions
6. **Visualizations**: 4 comprehensive plots
7. **Technical Details**: Response times, test coverage

## Visualization Types

1. **Overall Quality Trend**: Shows improvement over exchanges
2. **Performance Radar**: Multi-dimensional capability view
3. **Response Analysis**: Time distribution and length correlation
4. **Metric Trends**: Individual metric progression

## Cost Estimation

- Typical assessment: ~15-20 API calls
- Estimated cost: $0.05-0.15 per assessment
- Uses GPT-3.5-turbo for efficiency

## Interpreting Results

### Score Ratings
- ⭐ **Excellent** (8.0-10.0): Outstanding performance
- ✅ **Good** (7.0-7.9): Solid capabilities
- ⚠️ **Fair** (6.0-6.9): Adequate but needs work
- ❌ **Needs Work** (<6.0): Significant improvements needed

### Trend Indicators
- 📈 **Improving**: Positive slope >0.1
- ➡️ **Stable**: Slope between -0.1 and 0.1
- 📉 **Declining**: Negative slope <-0.1

## Advanced Usage

### Comparing Multiple Models

The tool can load and compare assessments:
```python
assessor = ConversationalAssessor()
assessments = [assess1, assess2, assess3]
comparison = assessor.compare_models(assessments)
```

### Custom Test Scenarios

Edit the `test_conversations` list in the assessor to add domain-specific tests.

## Troubleshooting

1. **API Key Error**: Ensure OPENAI_API_KEY is set
2. **Model Not Found**: Check model paths and LM head files exist
3. **Import Errors**: Install all required dependencies
4. **Plot Issues**: Matplotlib backend may need configuration

## Example Report Excerpt

```markdown
## Overall Performance Metrics

| Metric | Score | Trend | Rating |
|--------|-------|-------|--------|
| **Coherence** | 7.85/10 | 📈 Improving | ✅ Good |
| **Relevance** | 8.42/10 | ➡️ Stable | ⭐ Excellent |
| **Naturalness** | 7.23/10 | 📈 Improving | ✅ Good |
```

## Running Tests

```bash
python -m pytest test_assess_conversational.py -v
```

All 11 unit tests should pass, covering metrics calculation, evaluation, plotting, and report generation.