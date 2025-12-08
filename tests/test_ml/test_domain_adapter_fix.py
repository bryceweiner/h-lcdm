
import torch
import torch.nn as nn
import numpy as np
from pipeline.ml.domain_adapter import DomainAdaptationTrainer

# Mock classes to satisfy dependencies
class MockEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = {'mod1': nn.Linear(10, 5)}
    
    def encode(self, batch):
        return {'mod1': torch.randn(len(batch['mod1']), 5, requires_grad=True)}

def test_tensor_serialization_fix():
    # Setup
    base_model = MockEncoder()
    trainer = DomainAdaptationTrainer(base_model, latent_dim=5, n_surveys=2, device='cpu')
    
    # Create dummy batch
    batch = {'mod1': torch.randn(4, 10)}
    survey_ids = torch.tensor([0, 0, 1, 1])
    
    # Run adaptation step
    losses = trainer.adapt_domains(batch, survey_ids)
    
    # Verify losses are returned
    print("Losses returned:", losses)
    
    # Verify stored losses are floats, not tensors
    stored_losses = trainer.adaptation_losses[-1]
    print("Stored losses:", stored_losses)
    
    for k, v in stored_losses.items():
        if isinstance(v, torch.Tensor):
            print(f"FAIL: {k} is a Tensor: {v}")
            exit(1)
        if not isinstance(v, (float, int)):
            print(f"FAIL: {k} is not a number: {type(v)}")
            exit(1)
            
    print("SUCCESS: All stored losses are standard numbers.")

if __name__ == "__main__":
    test_tensor_serialization_fix()

