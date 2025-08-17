#!/usr/bin/env python3
"""
CLI interface for the Movie Scene Creator Multi-Agent System
"""
import argparse
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from graph import run_movie_creation, save_screenplay_files


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate a movie screenplay using AI multi-agent system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python cli.py \\
    --title "Neon Heist" \\
    --logline "A rookie hacker and a disillusioned cop team up for a one-night heist in a neon-soaked megacity." \\
    --genre "Cyberpunk thriller" \\
    --scenes 6
        """
    )
    
    parser.add_argument(
        "--title",
        required=True,
        help="Movie title"
    )
    
    parser.add_argument(
        "--logline",
        required=True,
        help="Brief one-sentence description of the movie"
    )
    
    parser.add_argument(
        "--genre",
        required=True,
        help="Movie genre (e.g., 'Cyberpunk thriller', 'Romantic comedy')"
    )
    
    parser.add_argument(
        "--scenes",
        type=int,
        default=5,
        help="Number of scenes to generate (default: 5)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Output directory for generated files (default: outputs)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Check if GROQ_API_KEY is set
    if not os.getenv("GROQ_API_KEY"):
        print("❌ Error: GROQ_API_KEY environment variable is not set.")
        print("Please set your Groq API key:")
        print("   export GROQ_API_KEY='your-api-key-here'")
        print("   # or add it to a .env file")
        sys.exit(1)
    
    try:
        print("🎬 Movie Scene Creator - Multi-Agent System")
        print("=" * 50)
        print(f"Title: {args.title}")
        print(f"Logline: {args.logline}")
        print(f"Genre: {args.genre}")
        print(f"Scenes: {args.scenes}")
        print("=" * 50)
        
        if args.verbose:
            print("Starting movie creation pipeline...")
        
        # Run the movie creation pipeline
        final_state = run_movie_creation(
            title=args.title,
            logline=args.logline,
            genre=args.genre,
            num_scenes=args.scenes
        )
        
        # Save the generated screenplay files
        fountain_path, markdown_path = save_screenplay_files(
            final_state, 
            output_dir=args.output_dir
        )
        
        print("\n🎉 Screenplay generation completed!")
        print(f"📄 Fountain format: {fountain_path}")
        print(f"📄 Markdown format: {markdown_path}")
        
        # Show some stats from the final state
        formatted_content = final_state.get("formatted_screenplay", {})
        if formatted_content:
            total_pages = formatted_content.get("total_estimated_pages", "Unknown")
            character_count = len(formatted_content.get("character_list", []))
            print(f"\n📊 Statistics:")
            print(f"   Estimated pages: {total_pages}")
            print(f"   Characters: {character_count}")
            
            # Show character list
            if character_count > 0:
                print(f"   Character list: {', '.join(formatted_content.get('character_list', []))}")
        
        if args.verbose:
            print(f"\n🔍 Pipeline stages completed:")
            stages = [
                ("Director Vision", "director_vision"),
                ("Scene Planning", "scene_plan"),
                ("Character Development", "characters"),
                ("Dialogue Writing", "dialogue"),
                ("Continuity Review", "continuity_review"),
                ("Formatting", "formatted_screenplay")
            ]
            
            for stage_name, state_key in stages:
                status = "✅" if final_state.get(state_key) else "❌"
                print(f"   {status} {stage_name}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Generation cancelled by user.")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Error during screenplay generation: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
