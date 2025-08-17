"""
Scene Planner Agent - Converts story beats into detailed scene outlines.
"""
import os
from langchain_groq.chat_models import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser


def run(state: dict) -> dict:
    """
    The Scene Planner agent converts beats into detailed scene structures.
    """
    # Initialize LLM using state parameters
    llm = ChatGroq(
        model_name=state.get("model_name", os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")),
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=state.get("temperature", float(os.getenv("GROQ_TEMPERATURE", "0.6")))
    )
    
    # Create parser
    parser = JsonOutputParser(pydantic_object=None)
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are an expert screenplay structure analyst. Your job is to convert story beats into detailed scene outlines.

For each beat provided, create a scene with:
- A proper slugline (location and time)
- A clear summary of what happens
- The scene's dramatic goal
- The central conflict or tension
- How the scene outcome advances the story

Return ONLY valid JSON matching this schema:
{{
  "scenes": [
    {{
      "slugline": "INT./EXT. LOCATION - TIME (e.g., 'INT. COFFEE SHOP - DAY')",
      "summary": "Detailed description of what happens in this scene",
      "goal": "What the protagonist wants to achieve in this scene",
      "conflict": "What opposes the protagonist or creates tension",
      "outcome": "How this scene pushes the story forward"
    }}
  ]
}}

IMPORTANT: Create scenes that directly serve the story beats. Be specific to the given genre, title, and logline.
"""),
        ("user", """
Movie: {title}
Logline: {logline}
Genre: {genre}

Story Beats to Convert:
{beats_text}

Convert each of these beats into a detailed scene outline. Make sure each scene serves the story and builds dramatic tension.
""")
    ])
    
    # Chain prompt, llm, and parser
    chain = prompt | llm | parser
    
    # Get beats from state and format them
    beats = state.get("beats", [])
    beats_text = "\n".join([f"{i+1}. {beat['name']}: {beat['what_happens']}" for i, beat in enumerate(beats)])
    
    if not beats_text:
        beats_text = "No beats provided - create generic scenes"
    
    # Get input variables from state
    input_vars = {
        "title": state.get('title', 'Untitled'),
        "logline": state.get('logline', 'A story unfolds'),
        "genre": state.get('genre', 'Drama'),
        "beats_text": beats_text
    }
    
    # Invoke the chain
    result = chain.invoke(input_vars)
    
    # Update state with scenes
    state["scenes"] = result.get("scenes", [])
    
    return state
