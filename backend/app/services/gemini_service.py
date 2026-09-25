"""
AI Service (Groq → Gemini → Ollama)
Integrates with cloud LLMs and falls back to a local Ollama instance
when the network is unavailable, API keys are exhausted, or rate limits hit.
"""
import os
import json
import time
import logging
from typing import Dict, Any, List, Optional
from urllib import request as urllib_request, error as urllib_error

try:
    from groq import Groq
except ImportError:
    Groq = None


class AIService:
    """
    Service for interacting with LLMs using a resilient fallback chain:
        1. Groq (cloud, fast)
        2. Gemini (cloud, Google)
        3. Ollama (local, offline-capable)

    Used for:
    - Analyzing datasets with natural language prompts to suggest target columns
    - Generating order reports based on predictions
    - AI-powered synthetic dataset generation
    """

    # Errors that should trigger a provider fallback
    _FALLBACK_SIGNALS = ('429', 'rate_limit', 'quota', 'timeout', 'connection',
                         'unreachable', 'refused', 'network', 'unavailable',
                         'exhausted', 'limit exceeded', 'apiconnection',
                         'no api key', 'invalid api key', 'authentication')

    def __init__(self):
        # Cloud credentials
        self.groq_key = os.environ.get('GROQ_API_KEY')
        self.gemini_key = os.environ.get('GEMINI_API_KEY')

        self.client = None
        self.provider = None
        self.gemini_model = os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash')

        # Ollama configuration (local fallback)
        self.ollama_url = os.environ.get('OLLAMA_URL', 'http://host.docker.internal:11434')
        self.ollama_model = os.environ.get('OLLAMA_MODEL', 'llama3.2')
        self._ollama_available = None  # lazy-checked

        # --- Attempt cloud providers first ---
        if self.groq_key and Groq:
            self.client = Groq(api_key=self.groq_key)
            self.provider = 'groq'
            self.model = 'llama-3.3-70b-versatile'
            print("✅ Using Groq API for AI analysis")
        elif self.gemini_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.gemini_key)
                self.client = genai.GenerativeModel(self.gemini_model)
                self.provider = 'gemini'
                print("✅ Using Gemini API for AI analysis")
            except Exception as e:
                print(f"⚠️ Failed to initialize Gemini: {e}")

        # If no cloud provider available, try Ollama as primary
        if not self.client:
            if self._check_ollama():
                self.provider = 'ollama'
                print(f"✅ Using local Ollama ({self.ollama_model}) for AI analysis — no cloud API keys found")
            else:
                print("⚠️ No AI provider available. Set GROQ_API_KEY, GEMINI_API_KEY, or run Ollama locally.")
                print("   AI features will return fallback results.")

    def _check_ollama(self) -> bool:
        """Check if Ollama is reachable."""
        if self._ollama_available is not None:
            return self._ollama_available
        try:
            req = urllib_request.Request(f'{self.ollama_url}/api/tags', method='GET')
            with urllib_request.urlopen(req, timeout=3) as resp:
                self._ollama_available = resp.status == 200
        except Exception:
            self._ollama_available = False
        return self._ollama_available

    def _is_fallback_error(self, error_msg: str) -> bool:
        """Check if an error should trigger provider fallback."""
        error_lower = error_msg.lower()
        return any(sig in error_lower for sig in self._FALLBACK_SIGNALS)

    def _call_ollama(self, system_prompt: str, user_message: str) -> str:
        """Call local Ollama instance via its HTTP API."""
        payload = json.dumps({
            'model': self.ollama_model,
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_message}
            ],
            'stream': False,
            'options': {
                'temperature': 0.3,
                'num_predict': 4000,
            }
        }).encode('utf-8')

        req = urllib_request.Request(
            f'{self.ollama_url}/api/chat',
            data=payload,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )

        try:
            with urllib_request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                content = data.get('message', {}).get('content', '')
                if not content:
                    raise ValueError('Ollama returned an empty response')
                return content
        except urllib_error.URLError as e:
            raise ValueError(f'Ollama unavailable at {self.ollama_url}: {e}') from e

    def call_llm(self, system_prompt: str, user_message: str) -> str:
        """Call the LLM with resilient fallback chain: Groq → Gemini → Ollama.

        Automatically falls through to the next provider when:
        - Rate limit (429) or quota exhausted
        - Network / connection error (no internet)
        - API key missing or invalid
        """
        last_error = None

        # ── 1. Try Groq ──
        if self.provider == 'groq' or (self.groq_key and Groq):
            try:
                if not self.client or self.provider != 'groq':
                    self.client = Groq(api_key=self.groq_key)
                    self.provider = 'groq'
                response = self.client.chat.completions.create(
                    model=self.model if hasattr(self, 'model') else 'llama-3.3-70b-versatile',
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=0.3,
                    max_tokens=4000
                )
                return response.choices[0].message.content
            except Exception as e:
                last_error = e
                err = str(e)
                if self._is_fallback_error(err):
                    print(f"⚠️ Groq failed ({err[:80]}…), falling back…", flush=True)
                else:
                    print(f"⚠️ Groq error: {err[:120]}", flush=True)

        # ── 2. Try Gemini ──
        if self.gemini_key:
            max_retries = 2
            retry_delay = 10
            for attempt in range(max_retries):
                try:
                    if self.provider != 'gemini':
                        import google.generativeai as genai
                        genai.configure(api_key=self.gemini_key)
                        self.client = genai.GenerativeModel(self.gemini_model)
                        self.provider = 'gemini'
                        print("↪ Switched to Gemini", flush=True)

                    response = self.client.generate_content([
                        {"role": "user", "parts": [system_prompt]},
                        {"role": "model", "parts": ["I understand. I will respond with valid JSON only."]},
                        {"role": "user", "parts": [user_message]}
                    ])

                    # Robustly extract text from Gemini response
                    try:
                        if response.text:
                            return response.text
                    except (ValueError, AttributeError):
                        if hasattr(response, 'candidates') and response.candidates:
                            candidate = response.candidates[0]
                            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                                parts = [p.text for p in candidate.content.parts if hasattr(p, 'text')]
                                if parts:
                                    return "".join(parts)
                        if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                            print(f"🚫 Gemini response blocked: {response.prompt_feedback}", flush=True)
                        raise ValueError("Gemini returned an empty response.")

                except Exception as e:
                    last_error = e
                    err = str(e)
                    if ('429' in err.lower() or 'quota' in err.lower()) and attempt < max_retries - 1:
                        print(f"⏳ Gemini quota hit, retrying in {retry_delay}s ({attempt+1}/{max_retries})…", flush=True)
                        time.sleep(retry_delay)
                        continue
                    print(f"⚠️ Gemini failed ({err[:80]}…), falling back to Ollama…", flush=True)
                    break  # move on to Ollama

        # ── 3. Try Ollama (local fallback) ──
        if self._check_ollama():
            try:
                if self.provider != 'ollama':
                    print(f"↪ Switched to local Ollama ({self.ollama_model})", flush=True)
                    self.provider = 'ollama'
                return self._call_ollama(system_prompt, user_message)
            except Exception as e:
                last_error = e
                print(f"⚠️ Ollama failed: {e}", flush=True)

        # ── All providers exhausted ──
        raise ValueError(
            f"All AI providers failed. Last error: {last_error}\n"
            f"Solutions:\n"
            f"  1. Check internet connectivity for Groq/Gemini\n"
            f"  2. Verify API keys in .env (GROQ_API_KEY, GEMINI_API_KEY)\n"
            f"  3. Run Ollama locally: 'ollama serve' then 'ollama pull {self.ollama_model}'"
        )

    def analyze_dataset_with_prompt(
        self,
        columns: List[str],
        column_types: Dict[str, str],
        sample_data: List[Dict[str, Any]],
        user_prompt: str
    ) -> Dict[str, Any]:
        """
        Analyze a dataset with a user's natural language prompt to suggest
        the best target column for prediction.
        """

        system_prompt = """You are an expert data scientist assistant. Your task is to analyze a dataset
and determine the best target column for machine learning based on the user's goal.

You must respond with valid JSON only, no markdown formatting. Use this exact structure:
{
    "suggested_target": "column_name",
    "problem_type": "classification|regression|timeseries",
    "reasoning": "Explanation of why this target column was chosen",
    "confidence": 0.85,
    "preprocessing_suggestions": [
        "suggestion 1",
        "suggestion 2"
    ],
    "environmental_factors": [
        "Any columns that appear to be environmental/external factors"
    ]
}"""

        # Format column info
        column_info = "\n".join([
            f"- {col} ({column_types.get(col, 'unknown')})"
            for col in columns
        ])

        # Format sample data
        sample_str = json.dumps(sample_data[:5], indent=2, default=str)

        user_message = f"""## User's Goal
{user_prompt}

## Dataset Columns
{column_info}

## Sample Data (first 5 rows)
{sample_str}

Based on the user's goal and the dataset structure, determine:
1. Which column should be the target for prediction
2. What type of ML problem this is (classification for categories, regression for numbers, timeseries for forecasting)
3. Your reasoning
4. Any preprocessing suggestions
5. Any columns that look like environmental/external factors (weather, holidays, etc.)

Respond with JSON only."""

        try:
            response_text = self.call_llm(system_prompt, user_message)

            # Clean up response if it has markdown code blocks
            response_text = response_text.strip()
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                # Remove first and last lines (```json and ```)
                response_text = "\n".join(lines[1:-1])

            result = json.loads(response_text)

            # Validate required fields
            required_fields = ['suggested_target', 'problem_type', 'reasoning']
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Missing required field: {field}")

            # Validate suggested target exists in columns
            if result['suggested_target'] not in columns:
                # Try to find a close match
                suggested = result['suggested_target'].lower()
                for col in columns:
                    if col.lower() == suggested or suggested in col.lower():
                        result['suggested_target'] = col
                        break

            return result

        except json.JSONDecodeError as e:
            return {
                'suggested_target': None,
                'problem_type': 'unknown',
                'reasoning': f'Failed to parse AI response: {str(e)}',
                'confidence': 0.0,
                'preprocessing_suggestions': [],
                'error': str(e)
            }
        except Exception as e:
            return {
                'suggested_target': None,
                'problem_type': 'unknown',
                'reasoning': f'AI API error: {str(e)}',
                'confidence': 0.0,
                'preprocessing_suggestions': [],
                'error': str(e)
            }

    def verify_problem_detection(
        self,
        columns: List[str],
        column_types: Dict[str, str],
        sample_data: List[Dict[str, Any]],
        target_column: Optional[str],
        deterministic_detection: Dict[str, Any],
        user_goal: str = ""
    ) -> Dict[str, Any]:
        """
        Ask AI to verify the deterministic problem-type decision.
        The caller should treat this as a second opinion, not the single source of truth.
        """

        system_prompt = """You are a senior machine learning reviewer. Verify a deterministic
problem-type decision for an AutoML training pipeline.

Respond with valid JSON only, no markdown. Use this exact structure:
{
    "agreement": true,
    "problem_type": "classification|binary_classification|multiclass_classification|regression|clustering|timeseries|object_detection|image_classification|unknown",
    "confidence": 0.85,
    "reasoning": "Short reason grounded in the target column and sample rows",
    "target_column": "column_name_or_null",
    "warnings": ["warning 1"],
    "recommended_algorithms": ["algorithm_key"]
}

Rules:
- Prefer the deterministic result when the evidence is ambiguous.
- Use classification for categorical labels, binary_classification for two labels, regression for continuous numeric targets.
- Use clustering only when there is no target column.
- Use timeseries only when the user's goal is forecasting or prediction over time.
- Never invent a target column that is not present."""

        sample_str = json.dumps(sample_data[:8], indent=2, default=str)
        detection_str = json.dumps(deterministic_detection, indent=2, default=str)
        column_info = "\n".join([
            f"- {col} ({column_types.get(col, 'unknown')})"
            for col in columns
        ])

        user_message = f"""## User Goal
{user_goal or 'Train the best model for the selected target.'}

## Selected Target
{target_column or 'None'}

## Dataset Columns
{column_info}

## Sample Rows
{sample_str}

## Deterministic Detection
{detection_str}

Verify whether the deterministic problem type is correct. Respond with JSON only."""

        try:
            response_text = self.call_llm(system_prompt, user_message).strip()
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                response_text = "\n".join(lines[1:-1])

            result = json.loads(response_text)
            result.setdefault('agreement', False)
            result.setdefault('confidence', 0.0)
            result.setdefault('warnings', [])
            result.setdefault('recommended_algorithms', [])

            if result.get('target_column') not in columns:
                result['target_column'] = target_column

            allowed_types = {
                'classification', 'binary_classification', 'multiclass_classification',
                'regression', 'clustering', 'timeseries', 'object_detection',
                'image_classification', 'unknown'
            }
            if result.get('problem_type') not in allowed_types:
                result['problem_type'] = 'unknown'
                result['confidence'] = 0.0

            return result
        except json.JSONDecodeError as e:
            return {
                'agreement': False,
                'problem_type': 'unknown',
                'confidence': 0.0,
                'reasoning': f'Failed to parse AI verification response: {str(e)}',
                'target_column': target_column,
                'warnings': [str(e)],
                'recommended_algorithms': []
            }
        except Exception as e:
            return {
                'agreement': False,
                'problem_type': 'unknown',
                'confidence': 0.0,
                'reasoning': f'AI verification unavailable: {str(e)}',
                'target_column': target_column,
                'warnings': [str(e)],
                'recommended_algorithms': []
            }

    # ── AI Synthetic Data Generation ──

    def generate_dataset_schema(self, prompt, num_columns=8):
        """Generate a dataset schema from a natural language description."""

        system_prompt = """You are an expert data engineer. Given a user's description of a dataset they need,
generate a complete schema with realistic column definitions.

You MUST respond with valid JSON only, no markdown. Use this exact structure:
{
    "name": "suggested_dataset_name",
    "description": "One-line description of the dataset",
    "columns": [
        {
            "name": "column_name_snake_case",
            "type": "numeric|integer|categorical|boolean|date|text|id",
            "description": "What this column represents",
            "sample_values": ["example1", "example2", "example3"],
            "config": {}
        }
    ]
}

Rules:
- Use snake_case for column names
- Choose realistic column types (not everything should be text)
- Include 1 ID column at the start
- sample_values should be realistic examples
- config should match the type:
  numeric/integer: {"min": N, "max": N}
  categorical: {"categories": ["val1", "val2", ...]}
  boolean: {"true_probability": 0.5}
  date: {}
  text: {"length": 20}
  id: {"prefix": "PREFIX"}
- Generate between 5 and 15 columns"""

        user_message = (
            f"Create a dataset schema for: {prompt}\n\n"
            f"Generate approximately {num_columns} columns. "
            f"Make columns realistic and domain-appropriate. Respond with JSON only."
        )

        try:
            response_text = self.call_llm(system_prompt, user_message)
            response_text = response_text.strip()
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                response_text = "\n".join(lines[1:-1])
            result = json.loads(response_text)
            if 'columns' not in result or not result['columns']:
                raise ValueError("No columns in response")
            return result
        except json.JSONDecodeError as e:
            return {'error': f'Failed to parse AI schema: {str(e)}'}
        except Exception as e:
            return {'error': f'AI schema generation failed: {str(e)}'}

    def generate_dataset_rows(self, schema, num_sample=30):
        """Generate realistic sample data rows from an approved schema."""

        col_descriptions = "\n".join([
            f"- {c['name']} ({c['type']}): {c.get('description', 'no description')}"
            for c in schema
        ])

        system_prompt = (
            "You are a data generator. Produce realistic sample data rows.\n"
            "Respond with valid JSON only — an array of objects. No markdown.\n\n"
            f"Column definitions:\n{col_descriptions}\n\n"
            "Rules:\n"
            "- Values must be realistic and internally consistent\n"
            "- Numeric values should be numbers not strings\n"
            "- Boolean values should be true/false not strings\n"
            "- Dates should be YYYY-MM-DD strings\n"
            "- IDs should be sequential with the column prefix\n"
            "- Create diversity — don't repeat the same values"
        )

        col_keys = ', '.join([c['name'] for c in schema])
        user_message = (
            f"Generate exactly {num_sample} realistic data rows as a JSON array.\n"
            f"Keys: {col_keys}\n"
            f"Respond with ONLY a JSON array."
        )

        try:
            response_text = self.call_llm(system_prompt, user_message)
            response_text = response_text.strip()
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                response_text = "\n".join(lines[1:-1])
            rows = json.loads(response_text)
            if not isinstance(rows, list):
                raise ValueError("Response is not an array")
            return rows
        except (json.JSONDecodeError, Exception):
            return []


# Singleton instance
_ai_service = None


def get_ai_service() -> AIService:
    """Get or create the AI service singleton"""
    global _ai_service
    if _ai_service is None:
        _ai_service = AIService()
    return _ai_service


# Backward compatibility aliases
GeminiService = AIService
get_gemini_service = get_ai_service
