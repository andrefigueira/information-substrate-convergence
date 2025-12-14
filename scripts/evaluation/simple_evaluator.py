#!/usr/bin/env python3
"""
Simple AI Conversation Evaluator
Works without complex dependencies
"""

import subprocess
import json
import csv
import os
from datetime import datetime

def test_conversation_quality():
    """Test AI conversational quality using simple subprocess calls"""

    test_questions = [
        "Hello, how are you today?",
        "What's your favorite hobby?",
        "Tell me about yourself.",
        "I'm feeling sad today. Can you help?",
        "What do you think about artificial intelligence?",
        "If you could travel anywhere, where would you go?",
        "How can I improve my communication skills?",
        "What makes a good friend?"
    ]

    print("🧪 Testing AI conversational quality...")

    results = []
    total_score = 0

    for i, question in enumerate(test_questions, 1):
        print(f"  [{i}/{len(test_questions)}] Testing: {question[:40]}...")

        try:
            # Run AI with question and capture response
            cmd = ['python', 'run_ai_chat.py']
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True)

            output, error = process.communicate(input=f"{question}\nquit\n", timeout=30)

            # Extract response from output
            lines = output.split('\n')
            response = ""
            for line in lines:
                if line.startswith('[ISC-AI]'):
                    response = line[9:].strip()
                    break

            if response:
                # Simple scoring
                score = score_response(question, response)
                total_score += score

                results.append({
                    'question': question,
                    'response': response,
                    'score': score
                })

                print(f"    Response: {response[:60]}...")
                print(f"    Score: {score:.2f}")
            else:
                print(f"    ⚠ No response captured")

        except Exception as e:
            print(f"    ⚠ Error: {e}")
            continue

    avg_score = total_score / len(results) if results else 0

    print(f"\n📊 RESULTS:")
    print(f"Tested: {len(results)}/{len(test_questions)} questions")
    print(f"Average Score: {avg_score:.3f}/1.0")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save to CSV
    with open(f"ai_evaluation_{timestamp}.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['question', 'response', 'score'])
        for result in results:
            writer.writerow([result['question'], result['response'], result['score']])

    print(f"📁 Results saved to ai_evaluation_{timestamp}.csv")

    return avg_score

def score_response(question, response):
    """Simple response scoring"""
    score = 0.0

    # Length check (not too short, not too long)
    words = len(response.split())
    if 10 <= words <= 100:
        score += 0.3
    elif 5 <= words <= 150:
        score += 0.2
    else:
        score += 0.1

    # Relevance check (contains words from question)
    question_words = set(question.lower().split())
    response_words = set(response.lower().split())
    overlap = len(question_words & response_words)
    if overlap > 0:
        score += min(0.3, overlap * 0.1)

    # Naturalness (avoid too much technical jargon)
    technical_terms = ['substrate', 'neuromorphic', 'phi', 'integration', 'concepts', 'nodes']
    tech_count = sum(1 for term in technical_terms if term in response.lower())
    if tech_count == 0:
        score += 0.3
    elif tech_count <= 2:
        score += 0.2
    else:
        score += 0.1

    # Engagement markers
    engagement_words = ['!', '?', 'you', 'think', 'feel', 'interesting', 'great']
    engagement_count = sum(1 for word in engagement_words if word in response.lower())
    if engagement_count > 0:
        score += min(0.1, engagement_count * 0.02)

    return min(1.0, score)

if __name__ == "__main__":
    test_conversation_quality()