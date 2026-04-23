"""
Skin Disease Detection Backend - Configuration Module
"""
from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Literal
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # App
    app_name: str = "Skin Disease Detection API"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Server
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    
    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@db:5432/skin_disease_db"
    
    # Model
    model_path: str = "ml/model_weights/efficientnetb3_skin_disease.keras"
    confidence_threshold: float = 0.5
    image_size: int = 300  # EfficientNet-B3 native input size
    
    # LLM Configuration
    # Providers: "gemini" (recommended), "openai"
    llm_provider: Literal["gemini", "openai"] = "gemini"
    
    # API keys
    google_api_key: str | None = None
    openai_api_key: str | None = None
    
    # Disease classes (in ALPHABETICAL order - matches training folder order)
    disease_classes: list[str] = [
        "Eczema",
        "Warts Molluscum and Viral Infections",
        "Melanoma", 
        "Atopic Dermatitis",
        "Basal Cell Carcinoma (BCC)",
        "Melanocytic Nevi (NV)",
        "Benign Keratosis-like Lesions (BKL)",
        "Psoriasis and Lichen Planus",
        "Seborrheic Keratoses and Benign Tumors",
        "Tinea Ringworm and Fungal Infections"
    ]
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "protected_namespaces": ("settings_",)
    }


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Disease information for LLM context
DISEASE_INFO = {
    "Eczema": {
        "severity": "mild-moderate",
        "contagious": False,
        "description": "A chronic inflammatory skin condition causing itchy, red, dry, and cracked skin."
    },
    "Melanoma": {
        "severity": "serious",
        "contagious": False,
        "description": "A serious form of skin cancer that develops from melanocytes. Early detection is critical."
    },
    "Atopic Dermatitis": {
        "severity": "mild-moderate",
        "contagious": False,
        "description": "A chronic form of eczema causing itchy, inflamed skin, often appearing in childhood."
    },
    "Basal Cell Carcinoma (BCC)": {
        "severity": "serious",
        "contagious": False,
        "description": "The most common type of skin cancer, usually caused by UV exposure. Generally slow-growing."
    },
    "Melanocytic Nevi (NV)": {
        "severity": "benign",
        "contagious": False,
        "description": "Common moles formed by clusters of melanocytes. Usually benign but should be monitored."
    },
    "Benign Keratosis-like Lesions (BKL)": {
        "severity": "benign",
        "contagious": False,
        "description": "Non-cancerous skin growths that may look like warts or moles. Generally harmless."
    },
    "Psoriasis and Lichen Planus": {
        "severity": "mild-moderate",
        "contagious": False,
        "description": "Autoimmune conditions causing scaly patches and skin inflammation."
    },
    "Seborrheic Keratoses and Benign Tumors": {
        "severity": "benign",
        "contagious": False,
        "description": "Non-cancerous skin growths that appear as waxy, raised spots. Common with age."
    },
    "Tinea Ringworm and Fungal Infections": {
        "severity": "mild",
        "contagious": True,
        "description": "Fungal infections causing ring-shaped, itchy, scaly patches on the skin."
    },
    "Warts Molluscum and Viral Infections": {
        "severity": "mild",
        "contagious": True,
        "description": "Viral skin infections causing small bumps or growths. Usually self-limiting."
    }
}
