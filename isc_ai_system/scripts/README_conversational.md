# ISC Conversational Model

This module enhances the ISC AI system with conversational capabilities while preserving its philosophical core.

## Setup

1. **Install Dependencies**:
```bash
pip install torch openai rich matplotlib numpy
```

2. **Set OpenAI API Key**:
You can set your OpenAI API key in one of two ways:
- Set the `OPENAI_API_KEY` environment variable:
  ```bash
  export OPENAI_API_KEY="your-api-key-here"
  ```
- Or edit the script and replace `YOUR_OPENAI_API_KEY` with your actual key

## Usage

### Running the Conversational Trainer

```bash
python conversational.py
```

The script offers two main options:

1. **Train an existing ISC model** to be more conversational
2. **Use an already trained conversational model** for chat

### Training a Model

When training, you'll be prompted to:
- Select an existing ISC model to enhance
- Set the number of parallel workers (default: 3, max: 8)
- Choose a training topic (e.g., "general conversation", "philosophy")
- Set the number of training exchanges (default: 30, max: 200)

Training will:
- Generate dialogue examples using GPT-3.5
- Enhance ISC responses to be more natural
- Train a conversational language model head
- Save checkpoints every 10 exchanges
- Generate progress plots and metrics

### Using a Trained Model for Chat

Once you have a trained conversational model, you can chat with it:
- Select option 2 from the main menu
- Choose your trained model from the list
- Type messages and receive conversational responses
- Type 'exit' to quit the chat

## Key Features

- **Preserves ISC Core**: The philosophical reasoning remains intact
- **Natural Conversations**: Responses are enhanced to sound more human-like
- **Context Awareness**: Maintains conversation history for coherent dialogue
- **Response Templates**: Uses templates for different conversation types
- **Progress Tracking**: Real-time metrics and visualization during training
- **Auto-saving**: Checkpoints saved automatically during training

## Architecture

The system consists of:

1. **ConversationalLMHead**: A neural network that converts ISC's concept vectors into natural language
2. **ConversationalTrainer**: Manages the training process with GPT-3.5 assistance
3. **ConversationalISC**: The chat interface for trained models

## Running Tests

```bash
python -m pytest test_conversational.py -v
```

All 24 unit tests should pass, covering:
- Data structures
- Neural network components
- Training functionality
- Chat interface
- Integration workflows

## Important Notes

- The API key issue has been fixed - it now uses environment variables
- The `process_input` compatibility issue with ISCCore has been handled with fallbacks
- Context vectors are properly batched for the attention mechanism
- The scheduler no longer uses the deprecated `verbose` parameter

## Troubleshooting

If you encounter issues:

1. **API Key Error**: Make sure your OpenAI API key is set correctly
2. **Model Loading Error**: Ensure the ISC model file exists and is compatible
3. **Memory Issues**: Reduce the number of parallel workers or training exchanges
4. **Import Errors**: Make sure you're running from the correct directory with proper paths

## Cost Estimation

The trainer tracks token usage and provides cost estimates:
- GPT-3.5-turbo pricing: $0.0005/1K prompt tokens, $0.0015/1K completion tokens
- Typical 30-exchange training session: ~$0.10-0.30
- Cost tracking is displayed in real-time during training