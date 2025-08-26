"""
Movie Scene Creator Multi-Agent System using LangGraph
"""
import os
import time
from typing import Dict, Any
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq

# Import all agents
from agents import director, scene_planner, character_dev, dialogue_writer, continuity_editor, formatter

# Load environment variables
load_dotenv()


# Timing wrapper functions for real-time measurement
def timed_agent_wrapper(agent_name: str, agent_func):
    """
    Wrapper function to measure actual execution time of each agent.
    """
    def wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
        # Initialize timing data in state if not present
        if 'agent_execution_times' not in state:
            state['agent_execution_times'] = {}
        if 'pipeline_start_time' not in state:
            state['pipeline_start_time'] = time.time()
        
        # Record start time for this agent
        agent_start_time = time.time()
        print(f"⏱️  Starting {agent_name} at {time.strftime('%H:%M:%S', time.localtime(agent_start_time))}...")
        
        # Execute the actual agent function
        result = agent_func(state)
        
        # Record end time and calculate duration
        agent_end_time = time.time()
        execution_time = agent_end_time - agent_start_time
        
        # Store timing information in the result state
        if isinstance(result, dict):
            if 'agent_execution_times' not in result:
                result['agent_execution_times'] = state.get('agent_execution_times', {})
            if 'pipeline_start_time' not in result:
                result['pipeline_start_time'] = state.get('pipeline_start_time', agent_start_time)
            
            # Record this agent's execution time
            result['agent_execution_times'][agent_name] = round(execution_time, 3)
            
            print(f"✅ {agent_name} completed in {execution_time:.3f} seconds")
        
        return result
    
    return wrapper


def calculate_total_execution_time(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate and store total pipeline execution time.
    """
    if 'pipeline_start_time' in state and 'agent_execution_times' in state:
        pipeline_end_time = time.time()
        total_time = pipeline_end_time - state['pipeline_start_time']
        
        # Store total execution time
        state['total_execution_time'] = round(total_time, 3)
        state['pipeline_end_time'] = pipeline_end_time
        
        # Calculate and verify sum of individual agent times
        sum_agent_times = sum(state['agent_execution_times'].values())
        
        print(f"\n⏱️  TIMING SUMMARY:")
        print(f"   Total Pipeline Time: {total_time:.3f} seconds")
        print(f"   Sum of Agent Times: {sum_agent_times:.3f} seconds")
        print(f"   Overhead Time: {(total_time - sum_agent_times):.3f} seconds")
        
        # Individual agent breakdown
        print(f"\n📊 AGENT EXECUTION BREAKDOWN:")
        for agent, exec_time in state['agent_execution_times'].items():
            percentage = (exec_time / total_time) * 100 if total_time > 0 else 0
            print(f"   {agent}: {exec_time:.3f}s ({percentage:.1f}%)")
    
    return state


def create_movie_graph() -> StateGraph:
    """
    Create and configure the LangGraph StateGraph for movie scene creation.
    """
    # Verify Groq API key is available
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY environment variable is required")
    
    # Test Groq connection
    test_llm = ChatGroq(
        model_name=os.getenv("GROQ_MODEL_NAME", "llama3-8b-8192"),
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=float(os.getenv("GROQ_TEMPERATURE", "0.1"))
    )
    
    # Define the state structure
    def create_initial_state(
        title: str,
        logline: str,
        genre: str,
        num_scenes: int = 5
    ) -> Dict[str, Any]:
        """Create initial state for the movie creation pipeline."""
        return {
            "title": title,
            "logline": logline,
            "genre": genre,
            "num_scenes": num_scenes,
            "director_vision": {},
            "scene_plan": {},
            "characters": {},
            "dialogue": {},
            "continuity_review": {},
            "formatted_screenplay": {}
        }
    
    # Create the StateGraph
    workflow = StateGraph(dict)
    
    # Add all agent nodes with their new roles, wrapped with timing measurement
    workflow.add_node("director", timed_agent_wrapper("director", director.run))  # Creates story beats
    workflow.add_node("scene_planner", timed_agent_wrapper("scene_planner", scene_planner.run))  # Converts beats to scenes
    workflow.add_node("character_dev", timed_agent_wrapper("character_dev", character_dev.run))  # Creates character profiles
    workflow.add_node("dialogue_writer", timed_agent_wrapper("dialogue_writer", dialogue_writer.run))  # Writes draft scenes
    workflow.add_node("continuity_editor", timed_agent_wrapper("continuity_editor", continuity_editor.run))  # Polishes final scenes
    workflow.add_node("formatter", timed_agent_wrapper("formatter", formatter.run))  # Formats to Fountain
    
    # Define the sequential workflow
    workflow.set_entry_point("director")
    workflow.add_edge("director", "scene_planner")
    workflow.add_edge("scene_planner", "character_dev")
    workflow.add_edge("character_dev", "dialogue_writer")
    workflow.add_edge("dialogue_writer", "continuity_editor")
    workflow.add_edge("continuity_editor", "formatter")
    workflow.add_edge("formatter", END)
    
    return workflow


def run_movie_creation(
    title: str,
    logline: str,
    genre: str,
    num_scenes: int = 5,
    model_name: str = None,
    temperature: float = None
) -> Dict[str, Any]:
    """
    Execute the complete movie creation pipeline.
    
    Args:
        title: Movie title
        logline: Brief story description
        genre: Movie genre
        num_scenes: Number of scenes to create
        model_name: Groq model to use (optional, defaults to env var)
        temperature: Temperature for AI generation (optional, defaults to env var)
    
    Returns:
        Final state containing all screenplay elements
    """
    print("🎬 Initializing Movie Scene Creator...")
    print(f"Title: {title}")
    print(f"Genre: {genre}")
    print(f"Scenes: {num_scenes}")
    print("-" * 50)
    
    # Create the workflow graph
    workflow = create_movie_graph()
    app = workflow.compile()
    
    # Create initial state with custom parameters
    initial_state = {
        "title": title,
        "logline": logline,
        "genre": genre,
        "num_scenes": num_scenes,
        "model_name": model_name or os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile"),
        "temperature": temperature if temperature is not None else float(os.getenv("GROQ_TEMPERATURE", "0.7")),
        "beats": [],
        "scenes": [],
        "characters": {},
        "draft_scenes": [],
        "final_scenes": [],
        "formatted_screenplay": {}
    }
    
    print("🎭 Starting Director Phase...")
    
    try:
        # Execute the workflow
        final_state = app.invoke(initial_state)
        
        # Calculate and display total execution time
        final_state = calculate_total_execution_time(final_state)
        
        print("✅ Movie creation pipeline completed successfully!")
        return final_state
        
    except Exception as e:
        print(f"❌ Error in movie creation pipeline: {str(e)}")
        raise e


def save_screenplay_files(state: Dict[str, Any], output_dir: str = "outputs") -> tuple:
    """
    Save the generated screenplay to Fountain and Markdown files.
    
    Args:
        state: Final state from the movie creation pipeline
        output_dir: Directory to save files
    
    Returns:
        Tuple of (fountain_file_path, markdown_file_path)
    """
    import os
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    formatted_content = state.get("formatted_screenplay", {})
    title = state.get("title", "untitled").lower().replace(" ", "_")
    
    # Save Fountain format
    fountain_content = formatted_content.get("fountain_screenplay", "No screenplay generated")
    fountain_path = os.path.join(output_dir, f"{title}.fountain")
    
    with open(fountain_path, "w", encoding="utf-8") as f:
        f.write(fountain_content)
    
    # Save Markdown format
    markdown_content = formatted_content.get("markdown_screenplay", "No screenplay generated")
    markdown_path = os.path.join(output_dir, f"{title}.md")
    
    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    
    return fountain_path, markdown_path


if __name__ == "__main__":
    # Test the system
    test_state = run_movie_creation(
        title="Test Movie",
        logline="A test movie to verify the system works.",
        genre="Drama",
        num_scenes=3
    )
    
    fountain_file, markdown_file = save_screenplay_files(test_state)
    print(f"\n📄 Files saved:")
    print(f"   Fountain: {fountain_file}")
    print(f"   Markdown: {markdown_file}")
