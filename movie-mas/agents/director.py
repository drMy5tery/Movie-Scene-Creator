"""
Director Agent - Establishes creative vision and breaks story into narrative beats.
"""
import os
from langchain_groq.chat_models import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser


def run(state: dict) -> dict:
    """
    The Director agent establishes creative vision and defines story beats.
    """
    # Initialize LLM with structured output using state parameters
    llm = ChatGroq(
        model_name=state.get("model_name", os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")), 
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=state.get("temperature", float(os.getenv("GROQ_TEMPERATURE", "0.7")))
    )
    
    # Define JSON schema for output
    output_schema = {
        "type": "object",
        "properties": {
            "beats": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Name of the story beat"},
                        "what_happens": {"type": "string", "description": "What happens in this beat"}
                    },
                    "required": ["name", "what_happens"]
                }
            }
        },
        "required": ["beats"]
    }
    
    # Create parser
    parser = JsonOutputParser(pydantic_object=None)
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are a visionary film director analyzing a movie concept and breaking it down into compelling story beats.

Given the title, logline, genre, and target number of scenes, create {num_scenes} distinct story beats that will drive the narrative forward. Each beat should be a major plot point or dramatic moment.

Analyze the story concept deeply:
- What are the key conflicts?
- What character transformations need to happen?
- What dramatic moments will engage the audience?
- How should tension build and release?

Return ONLY valid JSON matching this schema:
{{
  "beats": [
    {{
      "name": "Beat name (e.g., 'Opening Image', 'Inciting Incident', 'Midpoint Crisis')",
      "what_happens": "Detailed description of what occurs in this beat, specific to THIS story"
    }}
  ]
}}

IMPORTANT: Generate content specific to the given title, logline, and genre. Do not use generic placeholder text.
"""),
        ("user", """
Movie Details:
- Title: {title}
- Logline: {logline}
- Genre: {genre}
- Number of scenes needed: {num_scenes}

Create {num_scenes} compelling story beats that bring this specific story to life.
""")
    ])
    
    # Chain prompt, llm, and parser
    chain = prompt | llm | parser
    
    # Get input variables from state
    input_vars = {
        "title": state.get('title', 'Untitled'),
        "logline": state.get('logline', 'A story unfolds'),
        "genre": state.get('genre', 'Drama'),
        "num_scenes": state.get('num_scenes', 5)
    }
    
    # Invoke the chain
    result = chain.invoke(input_vars)
    
    # Update state with beats
    state["beats"] = result.get("beats", [])
    
    return state
