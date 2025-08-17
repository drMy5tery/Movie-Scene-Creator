#!/usr/bin/env python3
"""
Debug script to check what each agent produces
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_agents_step_by_step():
    """Test each agent individually to see where the issue occurs"""
    print("🔍 Debugging Agent Pipeline")
    print("=" * 50)
    
    # Initial state
    state = {
        "title": "Test Movie",
        "logline": "A simple test story for debugging.",
        "genre": "Drama",
        "num_scenes": 3,
        "beats": [],
        "scenes": [],
        "characters": {},
        "draft_scenes": [],
        "final_scenes": [],
        "formatted_screenplay": {}
    }
    
    try:
        # Test Director
        print("1️⃣ Testing Director Agent...")
        from agents.director import run
        state = run(state)
        print(f"   ✅ Beats created: {len(state.get('beats', []))}")
        if state.get('beats'):
            print(f"   📝 First beat: {state['beats'][0].get('name', 'Unknown')}")
    
        # Test Scene Planner  
        print("2️⃣ Testing Scene Planner Agent...")
        from agents.scene_planner import run
        state = run(state)
        print(f"   ✅ Scenes created: {len(state.get('scenes', []))}")
        if state.get('scenes'):
            print(f"   📝 First scene: {state['scenes'][0].get('slugline', 'Unknown')}")
        
        # Test Character Dev
        print("3️⃣ Testing Character Dev Agent...")  
        from agents.character_dev import run
        state = run(state)
        print(f"   ✅ Characters created: {len(state.get('characters', {}))}")
        if state.get('characters'):
            char_name = list(state['characters'].keys())[0]
            print(f"   📝 First character: {char_name}")
        
        # Test Dialogue Writer
        print("4️⃣ Testing Dialogue Writer Agent...")
        from agents.dialogue_writer import run
        state = run(state)
        print(f"   ✅ Draft scenes created: {len(state.get('draft_scenes', []))}")
        if state.get('draft_scenes'):
            print(f"   📝 First draft scene: {state['draft_scenes'][0].get('slugline', 'Unknown')}")
        
        # Test Continuity Editor
        print("5️⃣ Testing Continuity Editor Agent...")
        from agents.continuity_editor import run
        state = run(state)
        print(f"   ✅ Final scenes created: {len(state.get('final_scenes', []))}")
        if state.get('final_scenes'):
            print(f"   📝 First final scene: {state['final_scenes'][0].get('slugline', 'Unknown')}")
        
        # Test Formatter
        print("6️⃣ Testing Formatter Agent...")
        from agents.formatter import run
        state = run(state)
        formatted = state.get('formatted_screenplay', {})
        screenplay = formatted.get('fountain_screenplay', '')
        print(f"   ✅ Screenplay generated: {len(screenplay)} characters")
        print(f"   📝 Preview: {screenplay[:100]}...")
        
        print("\n🎉 All agents tested!")
        return state
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return state

if __name__ == "__main__":
    final_state = test_agents_step_by_step()
