"""
Dialogue Writer Agent - Writes screenplay scenes with dialogue and action.
"""
import os
from langchain_groq.chat_models import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser


def run(state: dict) -> dict:
    """
    The Dialogue Writer agent creates screenplay scenes with dialogue and action.
    """
    # Initialize LLM using state parameters
    llm = ChatGroq(
        model_name=state.get("model_name", os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")),
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=state.get("temperature", float(os.getenv("GROQ_TEMPERATURE", "0.8")))
    )
    
    # Create parser
    parser = JsonOutputParser(pydantic_object=None)
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are an expert screenplay writer. Your job is to write full screenplay scenes with dialogue and action lines.

For each scene outline provided, write a complete screenplay scene that includes:
- The slugline
- Action lines describing what we see
- Character dialogue that reflects their unique voice
- Proper screenplay formatting

Return ONLY valid JSON matching this schema:
{{
  "draft_scenes": [
    {{
      "slugline": "The scene's slugline (INT./EXT. LOCATION - TIME)",
      "content": "Complete screenplay scene with proper formatting, including action lines and dialogue"
    }}
  ]
}}

Screenplay formatting rules:
- Action lines in present tense, concise
- Character names in ALL CAPS when speaking
- Dialogue naturally flows and reveals character
- Each character has a distinct voice based on their profile
- Use consistent pronouns for each character (he/him, she/her, or they/them consistently)
- Avoid mixing singular and plural pronouns for the same character

IMPORTANT: Write specific, engaging scenes that serve the story. No placeholder text.
"""),
        ("user", """
Movie: {title}
Genre: {genre}

Character Profiles:
{characters_text}

Scenes to Write:
{scenes_text}

Write complete screenplay scenes for each outline. Make the dialogue natural and character-specific.
""")
    ])
    
    # Chain prompt, llm, and parser
    chain = prompt | llm | parser
    
    # Get scenes and characters from state and format them
    scenes = state.get("scenes", [])
    characters = state.get("characters", {})
    
    scenes_text = "\n".join([
        f"Scene: {scene['slugline']}\nGoal: {scene['goal']}\nConflict: {scene['conflict']}\nSummary: {scene['summary']}\n"
        for scene in scenes
    ])
    
    characters_text = "\n".join([
        f"{name}: {char['bio']} - Voice: {char['voice']} - Desires: {char['desires']}"
        for name, char in characters.items()
    ])
    
    if not scenes_text:
        scenes_text = "No scenes provided"
    if not characters_text:
        characters_text = "No characters provided"
    
    # Get input variables from state
    input_vars = {
        "title": state.get('title', 'Untitled'),
        "genre": state.get('genre', 'Drama'),
        "characters_text": characters_text,
        "scenes_text": scenes_text
    }
    
    # Invoke the chain
    result = chain.invoke(input_vars)
    
    # Update state with draft scenes
    state["draft_scenes"] = result.get("draft_scenes", [])
    
    return state
