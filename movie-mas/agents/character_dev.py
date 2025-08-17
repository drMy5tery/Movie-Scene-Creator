"""
Character Development Agent - Creates detailed character profiles based on scenes.
"""
import os
from langchain_groq.chat_models import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser


def run(state: dict) -> dict:
    """
    The Character Development agent creates character profiles based on scene requirements.
    """
    # Initialize LLM using state parameters
    llm = ChatGroq(
        model_name=state.get("model_name", os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")),
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=state.get("temperature", float(os.getenv("GROQ_TEMPERATURE", "0.7")))
    )
    
    # Create parser
    parser = JsonOutputParser(pydantic_object=None)
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are an expert character development specialist. Based on the movie concept and scene outlines, create detailed character profiles.

Analyze what characters are needed to serve the story:
- Who drives the main conflict?
- Who opposes the protagonist?
- What supporting characters are essential?
- How should each character speak and behave?

Return ONLY valid JSON matching this schema:
{{
  "characters": {{
    "CHARACTER_NAME": {{
      "bio": "Character background, age, occupation, key life experiences",
      "desires": "What they want most, their driving motivation",
      "voice": "How they speak - formal/casual, vocabulary, speech patterns",
      "quirks": "Unique behaviors, mannerisms, or traits that make them memorable"
    }}
  }}
}}

IMPORTANT: Create 3-5 essential characters that serve the specific story. Give each character a unique voice and personality.
"""),
        ("user", """
Movie: {title}
Logline: {logline}
Genre: {genre}

Scene Outlines:
{scenes_text}

Based on these scenes, create the essential characters needed to tell this story. Make each character distinct and purposeful.
""")
    ])
    
    # Chain prompt, llm, and parser
    chain = prompt | llm | parser
    
    # Get scenes from state and format them
    scenes = state.get("scenes", [])
    scenes_text = "\n".join([f"Scene: {scene['slugline']}\nSummary: {scene['summary']}\n" for scene in scenes])
    
    if not scenes_text:
        scenes_text = "No scenes provided - create generic characters"
    
    # Get input variables from state
    input_vars = {
        "title": state.get('title', 'Untitled'),
        "logline": state.get('logline', 'A story unfolds'),
        "genre": state.get('genre', 'Drama'),
        "scenes_text": scenes_text
    }
    
    # Invoke the chain
    result = chain.invoke(input_vars)
    
    # Update state with characters
    state["characters"] = result.get("characters", {})
    
    return state
