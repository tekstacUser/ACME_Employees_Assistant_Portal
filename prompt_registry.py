# 2.2 - Prompt Registry & A/B Testing
# Simple version management with A/B gate testing

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum

class PromptStatus(Enum):
    DRAFT = "draft"
    TEST = "test"
    STAGING = "staging"
    PRODUCTION = "production"

@dataclass
class PromptVersion:
    name: str
    version: int
    content: str
    status: PromptStatus = PromptStatus.DRAFT
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metrics: Dict[str, float] = field(default_factory=dict)

class MinimalPromptRegistry:
    """Minimal prompt registry with A/B testing"""

    def __init__(self):
        self.prompts: Dict[str, List[PromptVersion]] = {}
        # A/B testing gates
        self.gates = {
            "faithfulness": 0.85,
            "answer_relevance": 0.80,
            "latency_ms": 1000
        }

    def create_prompt(self, name: str, content: str) -> PromptVersion:
        """Create new prompt version"""
        if name not in self.prompts:
            self.prompts[name] = []
            version = 1
        else:
            version = len(self.prompts[name]) + 1

        prompt = PromptVersion(
            name=name,
            version=version,
            content=content,
            status=PromptStatus.DRAFT
        )
        self.prompts[name].append(prompt)
        return prompt

    def promote(self, name: str, version: int, metrics: Dict[str, float]) -> bool:
        """Check if metrics pass A/B gates"""
        for gate_name, threshold in self.gates.items():
            if gate_name in metrics:
                if metrics[gate_name] < threshold:
                    print(f"✗ Gate '{gate_name}' failed: {metrics[gate_name]:.2f} < {threshold}")
                    return False

        # All gates passed - promote
        for prompt in self.prompts[name]:
            if prompt.version == version:
                prompt.status = PromptStatus.PRODUCTION
                prompt.metrics = metrics
                print(f"✓ Promoted {name} v{version} to PRODUCTION")
                return True
        return False

    def get_current_production(self, name: str) -> Optional[PromptVersion]:
        """Get current production prompt"""
        prompts = self.prompts.get(name, [])
        for p in reversed(prompts):
            if p.status == PromptStatus.PRODUCTION:
                return p
        return None

    def get_all_versions(self, name: str) -> List[PromptVersion]:
        """Get all versions"""
        return self.prompts.get(name, [])


# Example usage
if __name__ == "__main__":
    registry = MinimalPromptRegistry()

    # Create v1
    v1 = registry.create_prompt("hr_assistant", "You are an HR assistant")
    print(f"Created: {v1.name} v{v1.version} - {v1.status.value}")

    # Create v2
    v2 = registry.create_prompt("hr_assistant", "You are a helpful HR assistant with expertise")
    print(f"Created: {v2.name} v{v2.version} - {v2.status.value}")

    # Test promotion with good metrics
    print("\nAttempting promotion with good metrics...")
    metrics = {
        "faithfulness": 0.92,
        "answer_relevance": 0.88,
        "latency_ms": 850
    }
    promoted = registry.promote("hr_assistant", 2, metrics)
    print(f"Promotion result: {promoted}")

    # Check production
    current = registry.get_current_production("hr_assistant")
    print(f"\nCurrent production: {current.name} v{current.version}")
