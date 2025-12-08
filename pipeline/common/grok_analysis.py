"""
Grok Analysis Client
===================

Client for interacting with the Grok API to generate qualitative analysis 
of ML pipeline results.
"""

import requests
import json
import logging
from typing import Dict, Any, Optional, List

class GrokAnalysisClient:
    """
    Client for generating qualitative analysis using Grok.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Grok client.
        
        Parameters:
            api_key: API key for x.ai. If not provided, will attempt to load from
                    XAI_API_KEY environment variable.
        """
        # Get API key from parameter, environment variable, or raise error
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.getenv('XAI_API_KEY')
            if not self.api_key:
                raise ValueError(
                    "xAI API key not provided. Either pass api_key parameter or "
                    "set XAI_API_KEY environment variable."
                )
        self.api_url = "https://api.x.ai/v1/chat/completions"
        self.logger = logging.getLogger(__name__)
        
    def generate_anomaly_report(self, 
                              anomalies: List[Dict[str, Any]], 
                              context: str = "cosmological data") -> str:
        """
        Generate a qualitative report for detected anomalies.
        
        Parameters:
            anomalies: List of anomaly dictionaries
            context: Context string for the analysis
            
        Returns:
            str: Generated analysis text
        """
        try:
            # Construct prompt
            prompt = self._construct_anomaly_prompt(anomalies, context)
            
            # Call API
            response = self._call_grok_api(prompt)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to generate Grok analysis: {e}")
            return f"Qualitative analysis unavailable: {e}"
    
    def _construct_anomaly_prompt(self, anomalies: List[Dict[str, Any]], context: str) -> str:
        """Construct the prompt for Grok."""
        
        # Format top anomalies for the prompt with context
        anomaly_lines = []
        for i, anomaly in enumerate(anomalies[:5]):  # Top 5
            score = anomaly.get('anomaly_score', 'N/A')
            idx = anomaly.get('sample_index', 'N/A')
            favored = anomaly.get('favored_model', 'INDETERMINATE')
            tags = anomaly.get('ontology_tags', [])
            ctx = anomaly.get('context', {}) if isinstance(anomaly.get('context'), dict) else {}
            zreg = ctx.get('redshift_regime', 'n/a')
            mods = ctx.get('modalities', [])
            anomaly_lines.append(
                f"- Anomaly {i+1} (idx={idx}, score={score}, model={favored}, tags={tags}, z={zreg}, mods={mods})"
            )
        anomaly_text = "\n".join(anomaly_lines) if anomaly_lines else "No anomalies provided."

        prompt = f"""
You are a senior theoretical physicist analyzing cosmological ML anomalies in {context}.

KEY DEFINITIONS:
- QTEP (Quantum Thermodynamic Entropy Partition): The ratio S_coh / |S_decoh| ≈ ln(2)/(1-ln(2)) ≈ 2.257, representing the fundamental information partition in quantum measurement.
- Gamma (γ): The holographic information processing rate (γ = H/π² in fundamental units) that serves as the affine parameter across all timelike worldlines.

The pipeline detected anomalies in a fused latent space (CMB, BAO, Voids, GW, Galaxies, JWST, FRB).
Ensemble detectors: Isolation Forest, HDBSCAN, VAE.

Top anomalies with context:
{anomaly_text}

Provide a concise scientific interpretation.
For each anomaly that survives statistical validation, use the specific format:
"Anomaly [X] was found in dataset [Y] corresponding to feature [Z]."

Then address:
- What do these high scores imply physically (new physics vs systematics vs rare objects)?
- How does the favored model (ΛCDM vs H-ΛCDM) influence interpretation?
- Which modalities/redshift regimes should be cross-checked next?
- Keep it rigorous and succinct.
"""
        
        return prompt

    def _call_grok_api(self, prompt: str) -> str:
        """Make the actual API call."""
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a senior theoretical physicist analyzing cosmological data."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "model": "grok-4-latest",
            "stream": False,
            "temperature": 0.2
        }
        
        response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        return data['choices'][0]['message']['content']

