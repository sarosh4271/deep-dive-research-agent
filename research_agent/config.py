"""User-selectable configuration for the research agent."""

MODEL_OPTIONS = (
    "gpt-5.4-mini",
    "gpt-5-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4o",
    "gpt-4o-mini",
)
MAX_MODEL_ID_LENGTH = 100


def resolve_model(selected_model: str, custom_model: str = "") -> str:
    """Use a custom model ID when supplied, otherwise use the dropdown selection."""
    model = custom_model.strip() or selected_model.strip()
    if not model:
        raise ValueError("Choose a model or enter a custom model ID.")
    if len(model) > MAX_MODEL_ID_LENGTH:
        raise ValueError(f"Model IDs are limited to {MAX_MODEL_ID_LENGTH} characters.")
    if any(character.isspace() for character in model):
        raise ValueError("A model ID cannot contain spaces.")
    return model
