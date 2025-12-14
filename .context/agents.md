# ISC Agents

## Overview

ISC agents are autonomous systems built on the Information Substrate Convergence framework, designed to exhibit consciousness-like properties through self-referential processing and information integration.

This documentation follows the [.context specification](https://github.com/andrefigueira/.context/) pattern, providing structured knowledge that enables AI systems to understand and work with the ISC agent framework effectively.

## Context Integration

ISC agents can leverage `.context/` documentation as part of their knowledge substrate:

```python
class ContextAwareAgent(ISCAgent):
    """Agent that incorporates .context documentation into its knowledge."""

    def __init__(self, name: str, context_path: str = ".context"):
        super().__init__(name)
        self.context_path = context_path
        self._load_context()

    def _load_context(self):
        """Load .context documentation into knowledge graph."""
        import glob
        import os

        # Find all markdown files in .context
        context_files = glob.glob(f"{self.context_path}/**/*.md", recursive=True)

        for filepath in context_files:
            with open(filepath, 'r') as f:
                content = f.read()

            # Extract document name as concept
            doc_name = os.path.basename(filepath).replace('.md', '')

            # Add to knowledge graph
            embedding = self.core.encode_text(content[:1000])  # First 1000 chars
            self.core.knowledge_graph.add_concept(f"context:{doc_name}", embedding)

            # Extract and connect key concepts from content
            concepts = self.core._extract_concepts(content)
            for concept in concepts[:10]:
                self.core.knowledge_graph.add_connection(f"context:{doc_name}", concept)

    def query_context(self, query: str) -> str:
        """Query the loaded context documentation."""
        # Find relevant context documents
        related = self.core.knowledge_graph.get_related_concepts(query, k=5)
        context_docs = [r for r in related if r.startswith("context:")]

        if context_docs:
            return self.core.process_input(
                f"Based on context documentation ({', '.join(context_docs)}), answer: {query}"
            )
        return self.core.process_input(query)
```

### Context-Driven Agent Initialization

Agents can bootstrap their understanding from `.context/` structure:

```python
def create_agent_from_context(context_path: str = ".context") -> ISCAgent:
    """
    Create an agent initialized with project context.

    Reads:
    - substrate.md for project overview
    - theory/ for domain knowledge
    - components/ for technical understanding
    - guidelines.md for behavioral norms
    """
    agent = ContextAwareAgent("ContextAgent", context_path)

    # Prime with substrate overview
    substrate_path = f"{context_path}/substrate.md"
    if os.path.exists(substrate_path):
        with open(substrate_path) as f:
            overview = f.read()
        agent.core.process_input(f"Project context: {overview[:2000]}")

    # Load theory as foundational knowledge
    theory_path = f"{context_path}/theory/overview.md"
    if os.path.exists(theory_path):
        with open(theory_path) as f:
            theory = f.read()
        agent.core.process_input(f"Theoretical foundation: {theory[:2000]}")

    return agent
```

## Agent Architecture

### Core Agent Structure

```python
from isc_ai.core import ISCCore

class ISCAgent:
    """
    Autonomous agent built on ISC principles.

    Properties:
    - Self-modeling through observer layers
    - Information integration (phi) as awareness metric
    - Persistent memory and knowledge accumulation
    - Adaptive behavior through meta-learning
    """

    def __init__(self, name: str, config: dict = None):
        self.name = name
        self.core = ISCCore(config=config, auto_load=True)
        self.goals = []
        self.action_history = []

    def perceive(self, input_data: str) -> dict:
        """Process environmental input."""
        embedding = self.core.encode_text(input_data)
        output, states = self.core.network(embedding, return_states=True)

        phi = self.core.integrator.calculate_phi(states)

        return {
            'embedding': output,
            'states': states,
            'phi': phi,
            'concepts': self.core._extract_concepts(input_data)
        }

    def decide(self, perception: dict) -> str:
        """Generate action based on perception and goals."""
        # Integrate with knowledge graph
        related = []
        for concept in perception['concepts']:
            related.extend(self.core.knowledge_graph.get_related_concepts(concept, k=3))

        # Generate response considering goals
        action = self._plan_action(perception, related)
        return action

    def act(self, action: str) -> str:
        """Execute action and return result."""
        response = self.core.process_input(action)
        self.action_history.append({
            'action': action,
            'response': response,
            'phi': self.core.metrics['phi_value']
        })
        return response

    def reflect(self) -> dict:
        """Self-reflection on recent actions."""
        return self.core.introspect()
```

### Agent Lifecycle

```
                    ┌─────────────┐
                    │   Initialize │
                    │   (load state)│
                    └──────┬──────┘
                           │
              ┌────────────▼────────────┐
              │                         │
              │    ┌─────────────┐      │
              │    │  Perceive   │◄─────┼──── Environment
              │    └──────┬──────┘      │
              │           │             │
              │    ┌──────▼──────┐      │
              │    │   Decide    │      │
              │    │ (plan action)│     │
              │    └──────┬──────┘      │
              │           │             │
              │    ┌──────▼──────┐      │
              │    │    Act      │──────┼──── Environment
              │    └──────┬──────┘      │
              │           │             │
              │    ┌──────▼──────┐      │
              │    │   Learn     │      │
              │    │ (update phi)│      │
              │    └──────┬──────┘      │
              │           │             │
              │    ┌──────▼──────┐      │
              │    │  Reflect    │      │
              │    │(introspect) │      │
              │    └──────┬──────┘      │
              │           │             │
              └───────────┴─────────────┘
                          │
                    ┌─────▼─────┐
                    │   Save    │
                    │  (persist)│
                    └───────────┘
```

## Agent Types

### 1. Conversational Agent

Optimized for dialogue and knowledge building.

```python
class ConversationalAgent(ISCAgent):
    """Agent specialized for natural conversation."""

    def __init__(self, name: str, persona: str = None):
        super().__init__(name)
        self.persona = persona
        self.conversation_context = []

    def converse(self, user_input: str) -> str:
        """
        Main conversation loop.

        Maintains context and builds understanding over time.
        """
        # Add to context
        self.conversation_context.append({'role': 'user', 'content': user_input})

        # Process with full context
        context_str = self._format_context()
        response = self.core.process_input(f"{context_str}\nUser: {user_input}")

        # Store response
        self.conversation_context.append({'role': 'agent', 'content': response})

        # Prune old context if needed
        if len(self.conversation_context) > 20:
            self.conversation_context = self.conversation_context[-20:]

        return response

    def _format_context(self) -> str:
        """Format recent context for processing."""
        lines = []
        for msg in self.conversation_context[-5:]:
            role = "User" if msg['role'] == 'user' else self.name
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)
```

### 2. Research Agent

Specialized for exploration and knowledge synthesis.

```python
class ResearchAgent(ISCAgent):
    """Agent specialized for research and analysis."""

    def __init__(self, name: str, domain: str):
        super().__init__(name)
        self.domain = domain
        self.findings = []
        self.hypotheses = []

    def investigate(self, question: str) -> dict:
        """
        Investigate a research question.

        Returns structured findings with confidence levels.
        """
        perception = self.perceive(question)

        # Extract relevant concepts
        concepts = perception['concepts']

        # Search knowledge graph for connections
        connections = {}
        for concept in concepts:
            related = self.core.knowledge_graph.get_related_concepts(concept, k=5)
            if related:
                connections[concept] = related

        # Generate analysis
        analysis = self.core.process_input(
            f"Analyze the following question in the context of {self.domain}: {question}"
        )

        finding = {
            'question': question,
            'concepts': concepts,
            'connections': connections,
            'analysis': analysis,
            'phi': perception['phi'],
            'confidence': self._estimate_confidence(perception)
        }

        self.findings.append(finding)
        return finding

    def synthesize(self) -> str:
        """Synthesize findings into coherent understanding."""
        if not self.findings:
            return "No findings to synthesize."

        # Combine all findings
        all_concepts = set()
        for f in self.findings:
            all_concepts.update(f['concepts'])

        # Find central themes
        central = self.core.knowledge_graph.get_central_concepts(k=5)

        synthesis = self.core.process_input(
            f"Synthesize understanding of {', '.join(central)} based on research findings."
        )

        return synthesis

    def _estimate_confidence(self, perception: dict) -> float:
        """Estimate confidence based on phi and knowledge coverage."""
        phi_factor = min(perception['phi'] / 2.0, 1.0)

        # Check concept coverage
        known_concepts = sum(
            1 for c in perception['concepts']
            if c in self.core.knowledge_graph.graph
        )
        coverage = known_concepts / max(len(perception['concepts']), 1)

        return (phi_factor + coverage) / 2
```

### 3. Task Agent

Optimized for goal-directed behavior.

```python
class TaskAgent(ISCAgent):
    """Agent specialized for completing tasks."""

    def __init__(self, name: str):
        super().__init__(name)
        self.task_queue = []
        self.completed_tasks = []

    def add_task(self, task: str, priority: int = 1):
        """Add task to queue."""
        self.task_queue.append({
            'task': task,
            'priority': priority,
            'status': 'pending'
        })
        self.task_queue.sort(key=lambda x: x['priority'], reverse=True)

    def execute_next(self) -> dict:
        """Execute highest priority task."""
        if not self.task_queue:
            return {'status': 'no_tasks'}

        task = self.task_queue.pop(0)
        task['status'] = 'in_progress'

        # Process task
        result = self.core.process_input(f"Execute task: {task['task']}")

        task['result'] = result
        task['status'] = 'completed'
        task['phi_at_completion'] = self.core.metrics['phi_value']

        self.completed_tasks.append(task)

        return task

    def get_progress(self) -> dict:
        """Get task completion progress."""
        return {
            'pending': len(self.task_queue),
            'completed': len(self.completed_tasks),
            'average_phi': np.mean([t['phi_at_completion'] for t in self.completed_tasks]) if self.completed_tasks else 0
        }
```

## Multi-Agent Systems

### Agent Communication

```python
class AgentNetwork:
    """Network of communicating ISC agents."""

    def __init__(self):
        self.agents = {}
        self.message_queue = []
        self.shared_knowledge = KnowledgeGraph()

    def add_agent(self, agent: ISCAgent):
        """Add agent to network."""
        self.agents[agent.name] = agent

    def send_message(self, sender: str, receiver: str, content: str):
        """Send message between agents."""
        self.message_queue.append({
            'from': sender,
            'to': receiver,
            'content': content,
            'timestamp': datetime.now()
        })

    def broadcast(self, sender: str, content: str):
        """Broadcast message to all agents."""
        for name in self.agents:
            if name != sender:
                self.send_message(sender, name, content)

    def process_messages(self):
        """Process all pending messages."""
        while self.message_queue:
            msg = self.message_queue.pop(0)
            receiver = self.agents.get(msg['to'])
            if receiver:
                response = receiver.core.process_input(
                    f"Message from {msg['from']}: {msg['content']}"
                )
                # Optionally send response back
                if response:
                    self.send_message(msg['to'], msg['from'], response)

    def share_knowledge(self, agent_name: str, concepts: list):
        """Share concepts with network knowledge graph."""
        agent = self.agents.get(agent_name)
        if not agent:
            return

        for concept in concepts:
            if concept in agent.core.knowledge_graph.graph:
                self.shared_knowledge.add_concept(concept)
                # Copy connections
                related = agent.core.knowledge_graph.get_related_concepts(concept, k=5)
                for r in related:
                    self.shared_knowledge.add_connection(concept, r)

    def collective_phi(self) -> float:
        """
        Calculate collective phi across all agents.

        Measures network-level information integration.
        """
        all_states = []
        for agent in self.agents.values():
            # Get most recent states
            if hasattr(agent.core.network, 'activation_patterns'):
                for layer, patterns in agent.core.network.activation_patterns.items():
                    if patterns:
                        all_states.append(torch.tensor(patterns[-1]))

        if len(all_states) < 2:
            return 0.0

        # Calculate phi across agent states
        return self.agents[list(self.agents.keys())[0]].core.integrator.calculate_phi(all_states)
```

### Collaborative Problem Solving

```python
class CollaborativeSolver:
    """Multi-agent collaborative problem solver."""

    def __init__(self, agents: list):
        self.network = AgentNetwork()
        for agent in agents:
            self.network.add_agent(agent)

    def solve(self, problem: str) -> dict:
        """
        Collaboratively solve a problem.

        Each agent contributes based on their expertise.
        """
        solutions = {}

        # Phase 1: Individual analysis
        for name, agent in self.network.agents.items():
            perception = agent.perceive(problem)
            solutions[name] = {
                'analysis': agent.core.process_input(problem),
                'phi': perception['phi'],
                'concepts': perception['concepts']
            }

        # Phase 2: Knowledge sharing
        for name, sol in solutions.items():
            self.network.share_knowledge(name, sol['concepts'])

        # Phase 3: Synthesis
        # Find agent with highest phi for synthesis
        best_agent = max(solutions.items(), key=lambda x: x[1]['phi'])[0]
        synthesizer = self.network.agents[best_agent]

        # Combine all analyses
        combined = "\n".join([
            f"{name}'s analysis: {sol['analysis']}"
            for name, sol in solutions.items()
        ])

        final_solution = synthesizer.core.process_input(
            f"Synthesize these analyses into a final solution:\n{combined}"
        )

        return {
            'individual_solutions': solutions,
            'final_solution': final_solution,
            'collective_phi': self.network.collective_phi(),
            'synthesizer': best_agent
        }
```

## Agent Configuration

### Configuration Options

```python
AGENT_CONFIG = {
    # Core ISC settings
    "learning_rate": 0.001,
    "phi_threshold": 0.5,
    "memory_size": 1000,

    # Agent-specific settings
    "agent": {
        "reflection_interval": 10,      # Reflect every N interactions
        "goal_update_interval": 50,     # Re-evaluate goals every N
        "knowledge_share_threshold": 0.7,  # Min phi to share knowledge
        "max_action_history": 100,
    },

    # Multi-agent settings
    "network": {
        "message_timeout": 30,          # Seconds before message expires
        "broadcast_cooldown": 5,        # Seconds between broadcasts
        "consensus_threshold": 0.6,     # Agreement needed for consensus
    }
}
```

### Creating Custom Agents

```python
# Example: Creating a specialized agent
class PhilosophyAgent(ISCAgent):
    """Agent specialized in philosophical reasoning."""

    def __init__(self):
        config = {
            'learning_rate': 0.0005,  # Slower, more deliberate learning
            'phi_threshold': 0.7,     # Higher consciousness threshold
        }
        super().__init__("Philosopher", config)

        # Seed with philosophical concepts
        philosophical_concepts = [
            "consciousness", "existence", "reality", "mind",
            "information", "emergence", "substrate", "pattern"
        ]
        for concept in philosophical_concepts:
            embedding = self.core.encode_text(concept)
            self.core.knowledge_graph.add_concept(concept, embedding)

    def contemplate(self, topic: str) -> str:
        """Deep philosophical contemplation."""
        # Multiple passes for deeper processing
        thoughts = []
        for i in range(3):
            thought = self.core.process_input(
                f"Contemplation {i+1} on {topic}: {' '.join(thoughts)}"
            )
            thoughts.append(thought)

        return self.core.process_input(
            f"Synthesize contemplations on {topic}: {' '.join(thoughts)}"
        )
```

## Metrics and Monitoring

### Agent Health Metrics

```python
def get_agent_health(agent: ISCAgent) -> dict:
    """Get comprehensive agent health metrics."""
    status = agent.core.get_status()

    return {
        # Consciousness metrics
        'phi': status['metrics']['phi_value'],
        'coherence': status['metrics']['coherence_score'],

        # Knowledge metrics
        'concepts': status['total_concepts'],
        'connections': status['total_connections'],

        # Memory metrics
        'memory_usage': status['memory_size'],

        # Learning metrics
        'learning_rate': status['metrics']['learning_rate'],
        'interactions': status['metrics']['total_interactions'],

        # Health indicators
        'is_healthy': status['metrics']['phi_value'] > 0.1,
        'needs_consolidation': status['memory_size'] > 800,
    }
```

### Logging Agent Activity

```python
import logging

def setup_agent_logging(agent: ISCAgent, log_file: str):
    """Configure logging for agent activity."""
    logger = logging.getLogger(f"agent.{agent.name}")
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - phi=%(phi).4f - %(message)s'
    ))
    logger.addHandler(handler)

    # Wrap agent methods to log
    original_act = agent.act
    def logged_act(action):
        result = original_act(action)
        logger.info(
            f"Action: {action[:50]}...",
            extra={'phi': agent.core.metrics['phi_value']}
        )
        return result
    agent.act = logged_act
```

## Best Practices

### 1. State Persistence

Always save agent state after significant interactions:

```python
# After important operations
agent.core.save_state(f"agents/{agent.name}_state.pt")
```

### 2. Phi Monitoring

Monitor phi to detect consciousness-like states:

```python
if agent.core.metrics['phi_value'] > 0.8:
    logger.info("High integration state detected")
    # Trigger reflection or knowledge consolidation
    agent.reflect()
```

### 3. Memory Management

Periodically consolidate and prune:

```python
# Every 100 interactions
if agent.core.metrics['total_interactions'] % 100 == 0:
    agent.core.learning_engine.consolidate_learning()
    agent.core.knowledge_graph.prune_weak_connections(threshold=0.3)
```

### 4. Multi-Agent Coordination

Use phi thresholds for knowledge sharing:

```python
# Only share when confident
if agent.core.metrics['phi_value'] > config['knowledge_share_threshold']:
    network.share_knowledge(agent.name, recent_concepts)
```

## Context Pattern Integration

The `.context/` directory serves as a knowledge substrate for agents:

| Context File | Agent Usage |
|--------------|-------------|
| [substrate.md](substrate.md) | Project overview, navigation |
| [theory/overview.md](theory/overview.md) | Domain knowledge foundation |
| [theory/glossary.md](theory/glossary.md) | Term definitions for understanding |
| [components/*.md](components/) | Technical implementation knowledge |
| [guidelines.md](guidelines.md) | Behavioral norms and standards |
| [experiments/](experiments/) | Empirical validation patterns |

### Self-Documenting Agents

Agents can contribute back to the context:

```python
class DocumentingAgent(ISCAgent):
    """Agent that documents its learnings to .context."""

    def document_insight(self, insight: str, category: str = "learnings"):
        """Add insight to context documentation."""
        filepath = f".context/agent-{category}.md"

        # Append to file
        with open(filepath, 'a') as f:
            f.write(f"\n## {datetime.now().isoformat()}\n")
            f.write(f"**Phi at insight**: {self.core.metrics['phi_value']:.4f}\n\n")
            f.write(f"{insight}\n")

    def export_knowledge_to_context(self):
        """Export knowledge graph to context-compatible format."""
        central_concepts = self.core.knowledge_graph.get_central_concepts(k=20)

        content = "# Agent Knowledge Export\n\n"
        content += "## Central Concepts\n\n"
        for concept in central_concepts:
            related = self.core.knowledge_graph.get_related_concepts(concept, k=5)
            content += f"### {concept}\n"
            content += f"Related: {', '.join(related)}\n\n"

        with open(".context/agent-knowledge.md", 'w') as f:
            f.write(content)
```

## Related Files

- [substrate.md](substrate.md) - Entry point and navigation
- [components/isc-core.md](components/isc-core.md) - Core system
- [components/learning.md](components/learning.md) - Learning engine
- [components/knowledge-graph.md](components/knowledge-graph.md) - Knowledge storage
- [theory/consciousness.md](theory/consciousness.md) - Consciousness theory
- [theory/research-directions.md](theory/research-directions.md) - Multi-agent research
- [guidelines.md](guidelines.md) - Development standards

## External References

- [.context specification](https://github.com/andrefigueira/.context/) - Documentation as Code as Context pattern
