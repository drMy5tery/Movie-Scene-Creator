"""
Continuity Editor Agent - Reviews and polishes draft scenes for consistency.
"""
import os
from langchain_groq.chat_models import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser


def run(state: dict) -> dict:
    """
    The Continuity Editor agent reviews draft scenes and creates final polished versions.
    """
    # Initialize LLM using state parameters
    llm = ChatGroq(
        model_name=state.get("model_name", os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")),
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=state.get("temperature", float(os.getenv("GROQ_TEMPERATURE", "0.4")))
    )
    
    # Create parser
    parser = JsonOutputParser(pydantic_object=None)
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are an expert screenplay continuity editor. Your job is to review draft scenes and create final polished versions.

Review each draft scene for:
- Character voice consistency
- Logical flow and pacing
- Clear scene transitions
- Proper screenplay formatting
- Engaging dialogue and action

Return ONLY valid JSON matching this schema:
{{
  "final_scenes": [
    {{
      "slugline": "The scene's slugline",
      "content": "Polished final screenplay scene with any necessary improvements"
    }}
  ]
}}

Make improvements to:
- Dialogue that feels unnatural
- Action lines that are unclear
- Character behavior that's inconsistent
- Pacing issues
- Formatting problems

IMPORTANT: Create final, production-ready screenplay scenes.
"""),
        ("user", """
Movie: {title}
Genre: {genre}

Character Profiles:
{characters_text}

Draft Scenes to Review:
{draft_scenes_text}

Review and polish each scene. Fix any issues with dialogue, pacing, or character consistency.
""")
    ])
    
    # Chain prompt, llm, and parser
    chain = prompt | llm | parser
    
    # Get draft scenes and characters from state and format them
    draft_scenes = state.get("draft_scenes", [])
    characters = state.get("characters", {})
    
    draft_scenes_text = "\n".join([
        f"Scene: {scene['slugline']}\n{scene['content']}\n---\n"
        for scene in draft_scenes
    ])
    
    characters_text = "\n".join([
        f"{name}: Voice - {char['voice']}, Quirks - {char['quirks']}"
        for name, char in characters.items()
    ])
    
    if not draft_scenes_text:
        draft_scenes_text = "No draft scenes provided"
    if not characters_text:
        characters_text = "No characters provided"
    
    # Get input variables from state
    input_vars = {
        "title": state.get('title', 'Untitled'),
        "genre": state.get('genre', 'Drama'),
        "characters_text": characters_text,
        "draft_scenes_text": draft_scenes_text
    }
    
    # Invoke the chain
    result = chain.invoke(input_vars)
    
    # Update state with final scenes
    state["final_scenes"] = result.get("final_scenes", [])
    
    return state
