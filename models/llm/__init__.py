"""LLM/VLM interfaces for scene understanding."""

from models.llm.schemas import (
    FloorSummaryOutput,
    FunctionalTag,
    ImageDescription,
    ObjectDescription,
    ObjectTag,
    RoomBrief,
    SceneObjectSummary,
    SceneSummary,
)
from models.llm.prompts import (
    system_instruction_floor_summary,
    system_instruction_functional_tagging,
    system_instruction_grounded_description,
    system_instruction_grounding,
    system_instruction_per_frame,
    system_instruction_summary,
    system_instruction_tagging,
)
try:
    from models.llm.openai_api import GPTInterface
except ImportError:
    GPTInterface = None

try:
    from models.llm.ollama_genai import OllamaGenAI, GenerationParams
except ImportError:
    OllamaGenAI = None
    GenerationParams = None

try:
    from models.llm.gpt_vlm import GPT_VLMInterface
except ImportError:
    GPT_VLMInterface = None

try:
    from models.llm.ollama_vlm import OllamaVLMInterface
except ImportError:
    OllamaVLMInterface = None

__all__ = [
    # Schemas
    "FloorSummaryOutput",
    "FunctionalTag",
    "ImageDescription",
    "ObjectDescription",
    "ObjectTag",
    "RoomBrief",
    "SceneObjectSummary",
    "SceneSummary",
    # Prompts
    "system_instruction_floor_summary",
    "system_instruction_functional_tagging",
    "system_instruction_grounded_description",
    "system_instruction_grounding",
    "system_instruction_per_frame",
    "system_instruction_summary",
    "system_instruction_tagging",
    # API Interfaces
    "GPTInterface",
    "OllamaGenAI",
    "GenerationParams",
    # VLM Interfaces
    "GPT_VLMInterface",
    "OllamaVLMInterface",
]
