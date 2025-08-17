#!/usr/bin/env python3
"""
Simple test script to verify agent functionality without running the full pipeline.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_director():
    """Test the director agent"""
    try:
        from agents.director import run
        
        test_state = {
            "title": "Neon Dreams",
            "logline": "A street artist discovers her graffiti comes to life in a dystopian city controlled by AI surveillance.",
            "genre": "Cyberpunk thriller",
            "num_scenes": 3
        }
        
        print("Testing Director Agent...")
        result = run(test_state)
        
        print("Director Result:")
        print(f"- Beats generated: {len(result.get('beats', []))}")
        for i, beat in enumerate(result.get('beats', [])[:2]):  # Show first 2 beats
            print(f"  {i+1}. {beat.get('name', 'Unknown')}: {beat.get('what_happens', 'Unknown')[:50]}...")
        
        return result
        
    except Exception as e:
        print(f"Director test failed: {e}")
        return None

def test_scene_planner(state):
    """Test the scene planner agent"""
    try:
        from agents.scene_planner import run
        
        print("\nTesting Scene Planner Agent...")
        result = run(state)
        
        print("Scene Planner Result:")
        print(f"- Scenes generated: {len(result.get('scenes', []))}")
        for i, scene in enumerate(result.get('scenes', [])[:2]):  # Show first 2 scenes
            print(f"  {i+1}. {scene.get('slugline', 'Unknown')}")
        
        return result
        
    except Exception as e:
        print(f"Scene planner test failed: {e}")
        return state

def test_character_dev(state):
    """Test the character development agent"""
    try:
        from agents.character_dev import run
        
        print("\nTesting Character Development Agent...")
        result = run(state)
        
        print("Character Development Result:")
        characters = result.get('characters', {})
        print(f"- Characters created: {len(characters)}")
        for name in list(characters.keys())[:2]:  # Show first 2 characters
            print(f"  - {name}: {characters[name].get('bio', 'Unknown')[:50]}...")
        
        return result
        
    except Exception as e:
        print(f"Character development test failed: {e}")
        return state

if __name__ == "__main__":
    print("🎬 Testing Movie Scene Creator Agents")
    print("=" * 50)
    
    # Check API key
    if not os.getenv("GROQ_API_KEY"):
        print("❌ GROQ_API_KEY not found in environment variables")
        exit(1)
    
    print(f"✅ Using model: {os.getenv('GROQ_MODEL_NAME', 'llama3-8b-8192')}")
    print(f"✅ Temperature: {os.getenv('GROQ_TEMPERATURE', '0.7')}")
    
    # Test each agent in sequence
    state = test_director()
    if state:
        state = test_scene_planner(state)
        state = test_character_dev(state)
    
    print("\n🎉 Agent testing completed!")
