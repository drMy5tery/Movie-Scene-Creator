#!/usr/bin/env python3
"""
Standalone Movie Scene Creator - Works without LangGraph
"""
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_story_beats(title, logline, genre, num_scenes):
    """Create story beats using direct Groq call"""
    try:
        from langchain_groq.chat_models import ChatGroq
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import JsonOutputParser
        
        llm = ChatGroq(
            model_name=os.getenv("GROQ_MODEL_NAME", "llama3-8b-8192"),
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.7
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a film director creating story beats. Return ONLY valid JSON:
{{
  "beats": [
    {{
      "name": "Beat name (e.g., 'Opening', 'Inciting Incident')",
      "what_happens": "Detailed description specific to THIS story"
    }}
  ]
}}"""),
            ("user", f"Create {num_scenes} story beats for:\nTitle: {title}\nLogline: {logline}\nGenre: {genre}")
        ])
        
        parser = JsonOutputParser()
        chain = prompt | llm | parser
        
        result = chain.invoke({})
        print(f"🎭 Director created {len(result.get('beats', []))} story beats")
        return result
        
    except Exception as e:
        print(f"❌ Director failed: {e}")
        return {"beats": [{"name": "Fallback", "what_happens": "Story unfolds"}]}

def create_scenes(title, logline, genre, beats):
    """Convert beats to scenes"""
    try:
        from langchain_groq.chat_models import ChatGroq
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import JsonOutputParser
        
        llm = ChatGroq(
            model_name=os.getenv("GROQ_MODEL_NAME", "llama3-8b-8192"),
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.6
        )
        
        beats_text = "\\n".join([f"{i+1}. {beat['name']}: {beat['what_happens']}" for i, beat in enumerate(beats)])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Convert story beats to scenes. Return ONLY valid JSON:
{{
  "scenes": [
    {{
      "slugline": "INT./EXT. LOCATION - TIME",
      "summary": "What happens in this scene",
      "goal": "Character's objective",
      "conflict": "What creates tension",
      "outcome": "How it advances the story"
    }}
  ]
}}"""),
            ("user", f"Convert these beats to scenes for {title} ({genre}):\\n{beats_text}")
        ])
        
        parser = JsonOutputParser()
        chain = prompt | llm | parser
        
        result = chain.invoke({})
        print(f"📝 Scene Planner created {len(result.get('scenes', []))} scenes")
        return result
        
    except Exception as e:
        print(f"❌ Scene Planner failed: {e}")
        return {"scenes": [{"slugline": "INT. LOCATION - DAY", "summary": "Scene happens", "goal": "TBD", "conflict": "TBD", "outcome": "TBD"}]}

def create_characters(title, genre, scenes):
    """Create characters based on scenes"""
    try:
        from langchain_groq.chat_models import ChatGroq
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import JsonOutputParser
        
        llm = ChatGroq(
            model_name=os.getenv("GROQ_MODEL_NAME", "llama3-8b-8192"),
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.7
        )
        
        scenes_text = "\\n".join([f"Scene: {scene['slugline']} - {scene['summary']}" for scene in scenes])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Create characters needed for the story. Return ONLY valid JSON:
{{
  "characters": {{
    "CHARACTER_NAME": {{
      "bio": "Background, age, occupation",
      "desires": "What they want most",
      "voice": "How they speak",
      "quirks": "Unique traits"
    }}
  }}
}}"""),
            ("user", f"Create 3-4 characters for {title} ({genre}) based on these scenes:\\n{scenes_text}")
        ])
        
        parser = JsonOutputParser()
        chain = prompt | llm | parser
        
        result = chain.invoke({})
        print(f"👥 Character Dev created {len(result.get('characters', {}))} characters")
        return result
        
    except Exception as e:
        print(f"❌ Character Dev failed: {e}")
        return {"characters": {"PROTAGONIST": {"bio": "Main character", "desires": "To succeed", "voice": "Determined", "quirks": "Brave"}}}

def write_screenplay(title, genre, scenes, characters):
    """Write the final screenplay"""
    try:
        from langchain_groq.chat_models import ChatGroq
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import JsonOutputParser
        
        llm = ChatGroq(
            model_name=os.getenv("GROQ_MODEL_NAME", "llama3-8b-8192"),
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.8
        )
        
        scenes_text = "\\n".join([f"Scene: {scene['slugline']}\\nSummary: {scene['summary']}\\nGoal: {scene['goal']}" for scene in scenes])
        char_text = "\\n".join([f"{name}: {char['bio']} - Voice: {char['voice']}" for name, char in characters.items()])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Write a complete screenplay. Return ONLY valid JSON:
{{
  "screenplay": "Complete screenplay in Fountain format with proper dialogue and action"
}}"""),
            ("user", f"Write a complete screenplay for {title} ({genre}):\\n\\nCharacters:\\n{char_text}\\n\\nScenes:\\n{scenes_text}")
        ])
        
        parser = JsonOutputParser()
        chain = prompt | llm | parser
        
        result = chain.invoke({})
        print(f"📄 Screenplay written!")
        return result.get("screenplay", f"Title: {title}\\n\\nFADE IN:\\n\\n[Screenplay content here]\\n\\nFADE OUT.")
        
    except Exception as e:
        print(f"❌ Screenplay writing failed: {e}")
        return f"Title: {title}\\n\\nFADE IN:\\n\\n[Error generating screenplay]\\n\\nFADE OUT."

def main():
    """Main function"""
    print("🎬 Standalone Movie Scene Creator")
    print("=" * 40)
    
    if not os.getenv("GROQ_API_KEY"):
        print("❌ GROQ_API_KEY not found in .env file")
        return
    
    # Test input
    title = "Neon Dreams"
    logline = "A street artist discovers her graffiti comes to life in a dystopian city controlled by AI surveillance."
    genre = "Cyberpunk thriller"
    num_scenes = 3
    
    print(f"Title: {title}")
    print(f"Genre: {genre}")
    print(f"Scenes: {num_scenes}")
    print("-" * 40)
    
    # Run the pipeline
    beats_result = create_story_beats(title, logline, genre, num_scenes)
    scenes_result = create_scenes(title, logline, genre, beats_result.get("beats", []))
    characters_result = create_characters(title, genre, scenes_result.get("scenes", []))
    screenplay = write_screenplay(title, genre, scenes_result.get("scenes", []), characters_result.get("characters", {}))
    
    print("\\n🎉 Generation complete!")
    print("\\n📄 Screenplay Preview:")
    print("-" * 40)
    print(screenplay[:500] + "..." if len(screenplay) > 500 else screenplay)
    
    # Save to file
    filename = f"outputs/{title.lower().replace(' ', '_')}.fountain"
    os.makedirs("outputs", exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(screenplay)
    print(f"\\n✅ Saved to: {filename}")

if __name__ == "__main__":
    main()
