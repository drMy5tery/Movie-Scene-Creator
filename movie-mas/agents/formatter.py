"""
Formatter Agent - Converts all screenplay content into proper Fountain format.
"""
import os
import json
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage


def run(state: dict) -> dict:
    """
    The Formatter agent converts all screenplay content into proper Fountain format.
    """
    # Gather all screenplay elements
    title = state.get('title', 'Untitled')
    logline = state.get('logline', 'A story unfolds')
    genre = state.get('genre', 'Drama')
    characters = state.get('characters', {})
    final_scenes = state.get('final_scenes', [])
    
    # Build the screenplay from the final scenes content
    if final_scenes:
        # Use the actual generated content from the final scenes
        screenplay_content = f"Title: {title}\n\n{logline}\n\nFADE IN:\n\n"
        
        for i, scene in enumerate(final_scenes):
            scene_content = scene.get('content', '')
            if scene_content:
                screenplay_content += scene_content
                # Add transitions between scenes (except for the last one)
                if i < len(final_scenes) - 1:
                    screenplay_content += "\n\nCUT TO:\n\n"
                else:
                    screenplay_content += "\n\nFADE OUT.\n\nTHE END"
            else:
                # Fallback if no content
                slugline = scene.get('slugline', f'INT. LOCATION - DAY')
                screenplay_content += f"{slugline}\n\nScene content here.\n\n"
        
        # Extract character names for the character list
        character_list = list(characters.keys()) if characters else ["Various Characters"]
        
        formatted_output = {
            "fountain_screenplay": screenplay_content,
            "character_list": character_list,
            "scene_breakdown": [f"Scene {i+1}: {scene.get('slugline', 'Unknown')}" for i, scene in enumerate(final_scenes)],
            "total_estimated_pages": max(10, len(final_scenes) * 3),
            "formatting_notes": "Formatted from final scenes content"
        }
    else:
        # Fallback if no final scenes
        fountain_content = f"""Title: {title}

{logline}

FADE IN:

INT. LOCATION - DAY

No scenes were generated.

FADE OUT.

THE END"""
        
        formatted_output = {
            "fountain_screenplay": fountain_content,
            "character_list": list(characters.keys()) if characters else ["No characters"],
            "scene_breakdown": ["No scenes generated"],
            "total_estimated_pages": 1,
            "formatting_notes": "Fallback content - no final scenes available"
        }
    
    # Also create a Markdown version for easier reading
    fountain_screenplay = formatted_output.get("fountain_screenplay", "No screenplay content generated")
    
    markdown_content = f"""# {state.get('title', 'Untitled')}

## Logline
{state.get('logline', 'No logline provided')}

## Genre
{state.get('genre', 'Drama')}

## Characters
"""
    
    # Add character information to markdown
    for name, char_info in characters.items():
        markdown_content += f"\n### {name}\n"
        markdown_content += f"- **Bio:** {char_info.get('bio', 'Unknown')}\n"
        markdown_content += f"- **Desires:** {char_info.get('desires', 'Unknown')}\n"
        markdown_content += f"- **Voice:** {char_info.get('voice', 'Unknown')}\n"
    
    markdown_content += f"\n\n## Screenplay\n\n```fountain\n{fountain_screenplay}\n```"
    
    formatted_output["markdown_screenplay"] = markdown_content
    
    # Update state with formatted content
    state["formatted_screenplay"] = formatted_output
    
    return state
