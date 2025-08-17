#!/usr/bin/env python3
"""
Inspect the full state to see what data is available
"""
import os
import json
from dotenv import load_dotenv

load_dotenv()

def inspect_full_state():
    """Run all agents and inspect the final state"""
    
    # Initial state
    state = {
        "title": "Test Movie",
        "logline": "A simple test story for debugging.",
        "genre": "Drama", 
        "num_scenes": 3
    }
    
    # Run all agents
    from agents.director import run as director_run
    from agents.scene_planner import run as scene_planner_run
    from agents.character_dev import run as character_dev_run
    from agents.dialogue_writer import run as dialogue_writer_run
    from agents.continuity_editor import run as continuity_editor_run
    
    print("🔍 Running full pipeline and inspecting state...")
    
    state = director_run(state)
    state = scene_planner_run(state)
    state = character_dev_run(state)
    state = dialogue_writer_run(state)
    state = continuity_editor_run(state)
    
    print("\n📊 Final State Analysis:")
    print("=" * 50)
    
    for key, value in state.items():
        if isinstance(value, list):
            print(f"📝 {key}: {len(value)} items")
            if value:
                print(f"   First item keys: {list(value[0].keys()) if isinstance(value[0], dict) else 'Not a dict'}")
        elif isinstance(value, dict):
            print(f"📝 {key}: {len(value)} keys")
            if value:
                print(f"   Keys: {list(value.keys())}")
        else:
            print(f"📝 {key}: {type(value).__name__} = {str(value)[:50]}...")
    
    # Check final scenes content specifically
    if 'final_scenes' in state and state['final_scenes']:
        print(f"\n🎬 Final Scenes Content:")
        for i, scene in enumerate(state['final_scenes'][:2]):  # Show first 2 scenes
            print(f"Scene {i+1}:")
            print(f"  Slugline: {scene.get('slugline', 'None')}")
            content = scene.get('content', '')
            print(f"  Content: {content[:100]}..." if len(content) > 100 else f"  Content: {content}")
            print()

if __name__ == "__main__":
    inspect_full_state()
