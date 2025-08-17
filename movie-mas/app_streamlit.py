"""
Streamlit Web Interface for the Movie Scene Creator Multi-Agent System
"""
import streamlit as st
import os
import sys
from pathlib import Path
import tempfile
import zipfile
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from graph import run_movie_creation, save_screenplay_files


def process_screenplay_content(content):
    """
    Process screenplay markdown content for better display in Streamlit.
    Handles line wrapping, formatting, and proper HTML escaping.
    """
    import html
    import re
    
    # Escape HTML characters to prevent issues
    content = html.escape(content)
    
    # Convert markdown headers to proper HTML
    content = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', content, flags=re.MULTILINE)
    content = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', content, flags=re.MULTILINE)
    content = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', content, flags=re.MULTILINE)
    
    # Convert **bold** to <strong>
    content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', content)
    
    # Convert *italic* to <em>
    content = re.sub(r'\*(.*?)\*', r'<em>\1</em>', content)
    
    # Convert line breaks to proper HTML
    content = content.replace('\n\n', '</p><p>')
    content = content.replace('\n', '<br>')
    
    # Wrap in paragraph tags if not already wrapped
    if not content.startswith('<'):
        content = f'<p>{content}</p>'
    
    # Fix any double paragraph tags
    content = re.sub(r'<p></p>', '', content)
    content = re.sub(r'<p><h([1-3])>', r'<h\1>', content)
    content = re.sub(r'</h([1-3])></p>', r'</h\1>', content)
    
    return content


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Movie Scene Creator",
        page_icon="🎬",
        layout="wide"
    )
    
    st.title("🎬 Movie Scene Creator")
    st.subheader("AI Multi-Agent System for Screenplay Generation")
    
    # Check if API key is configured in .env file
    if not os.getenv("GROQ_API_KEY"):
        st.error("⚠️ GROQ_API_KEY not found in environment variables.")
        st.info("""
        **Setup Required:**
        1. Get a free API key from [Groq Console](https://console.groq.com/keys)
        2. Add it to your `.env` file:
           ```
           GROQ_API_KEY=your-api-key-here
           GROQ_MODEL_NAME=llama-3.3-70b-versatile
           GROQ_TEMPERATURE=0.7
           ```
        3. Restart the Streamlit app
        """)
        return
    
    # Sidebar for system configuration
    with st.sidebar:
        st.header("⚙️ System Configuration")
        
        # Available Groq models (updated from current Groq API)
        # Production Models (recommended for production use)
        production_models = {
            "llama-3.3-70b-versatile": {
                "name": "Llama 3.3 70B Versatile",
                "description": "Best for creative content (Default) - Production",
                "speed": "Medium",
                "quality": "Excellent",
                "context": "131K",
                "developer": "Meta",
                "type": "Production"
            },
            "llama-3.1-8b-instant": {
                "name": "Llama 3.1 8B Instant",
                "description": "Fastest response times - Production",
                "speed": "Very Fast",
                "quality": "Good",
                "context": "131K",
                "developer": "Meta",
                "type": "Production"
            },
            "meta-llama/llama-guard-4-12b": {
                "name": "Llama Guard 4 12B",
                "description": "Content safety and moderation - Production",
                "speed": "Fast",
                "quality": "Very Good",
                "context": "131K",
                "developer": "Meta",
                "type": "Production"
            }
        }
        
        # Preview Models (evaluation purposes only)
        preview_models = {
            "deepseek-r1-distill-llama-70b": {
                "name": "DeepSeek R1 Distill Llama 70B",
                "description": "Advanced reasoning capabilities - Preview",
                "speed": "Medium",
                "quality": "Excellent",
                "context": "131K",
                "developer": "DeepSeek/Meta",
                "type": "Preview"
            },
            "meta-llama/llama-4-maverick-17b-128e-instruct": {
                "name": "Llama 4 Maverick 17B",
                "description": "Next-gen Llama model - Preview",
                "speed": "Fast",
                "quality": "Very Good",
                "context": "131K",
                "developer": "Meta",
                "type": "Preview"
            },
            "meta-llama/llama-4-scout-17b-16e-instruct": {
                "name": "Llama 4 Scout 17B",
                "description": "Efficient next-gen model - Preview",
                "speed": "Fast",
                "quality": "Very Good",
                "context": "131K",
                "developer": "Meta",
                "type": "Preview"
            },
            "openai/gpt-oss-120b": {
                "name": "GPT OSS 120B",
                "description": "OpenAI's open source model (Large) - Preview",
                "speed": "Slow",
                "quality": "Excellent",
                "context": "131K",
                "developer": "OpenAI",
                "type": "Preview"
            },
            "openai/gpt-oss-20b": {
                "name": "GPT OSS 20B",
                "description": "OpenAI's open source model (Medium) - Preview",
                "speed": "Fast",
                "quality": "Very Good",
                "context": "131K",
                "developer": "OpenAI",
                "type": "Preview"
            },
            "qwen/qwen3-32b": {
                "name": "Qwen 3 32B",
                "description": "Alibaba's multilingual model - Preview",
                "speed": "Fast",
                "quality": "Very Good",
                "context": "131K",
                "developer": "Alibaba Cloud",
                "type": "Preview"
            },
            "moonshotai/kimi-k2-instruct": {
                "name": "Kimi K2 Instruct",
                "description": "Moonshot AI's instruction model - Preview",
                "speed": "Fast",
                "quality": "Very Good",
                "context": "131K",
                "developer": "Moonshot AI",
                "type": "Preview"
            }
        }
        
        # Combine all models (production first, then preview)
        available_models = {**production_models, **preview_models}
        
        # Get default values from environment or set defaults
        default_model = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
        default_temp = float(os.getenv("GROQ_TEMPERATURE", "0.7"))
        
        # Model selection
        st.subheader("🤖 AI Model Selection")
        
        # Create a list of model options for the selectbox
        model_options = list(available_models.keys())
        model_names = [available_models[key]["name"] for key in model_options]
        
        # Find the index of the default model
        try:
            default_index = model_options.index(default_model)
        except ValueError:
            default_index = 0
        
        selected_model_name = st.selectbox(
            "Choose AI Model",
            model_names,
            index=default_index,
            help="Select the Groq model to use for screenplay generation"
        )
        
        # Get the actual model key from the selected name
        selected_model = model_options[model_names.index(selected_model_name)]
        
        # Show model details
        model_info = available_models[selected_model]
        model_type_color = "🟢" if model_info['type'] == 'Production' else "🟡"
        st.info(f"""
        **{model_info['name']}** {model_type_color}
        - {model_info['description']}
        - Developer: {model_info['developer']}
        - Speed: {model_info['speed']}
        - Quality: {model_info['quality']}
        - Context: {model_info['context']} tokens
        """)
        
        # Temperature control
        st.subheader("🌡️ Creativity Control")
        
        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=1.0,
            value=default_temp,
            step=0.1,
            help="Controls creativity vs consistency. Lower = more consistent, Higher = more creative"
        )
        
        # Temperature guidance
        if temperature <= 0.4:
            temp_desc = "🎯 Focused & Consistent"
            temp_color = "🔵"
        elif temperature <= 0.7:
            temp_desc = "⚖️ Balanced"
            temp_color = "🟢"
        else:
            temp_desc = "🎨 Creative & Varied"
            temp_color = "🟠"
        
        st.markdown(f"{temp_color} **{temp_desc}**")
        
        # Store selected values in session state
        st.session_state.selected_model = selected_model
        st.session_state.selected_temperature = temperature
        
        st.markdown("---")
        
        # Current configuration summary
        st.subheader("📋 Current Settings")
        st.code(f"""
Model: {selected_model}
Temperature: {temperature}
API: Groq (configured ✅)
""")
        
        st.markdown("---")
        st.markdown("""
        **About this system:**
        - 🤖 Uses LangGraph for agent orchestration
        - ⚡ Powered by Groq's fast inference  
        - 📄 Generates professional screenplay formats
        - 🎭 Six specialized AI agents working together
        """)
    
    # Main input form
    st.header("📝 Movie Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        title = st.text_input(
            "Movie Title *",
            value=st.session_state.get('title', ''),
            placeholder="e.g., Neon Heist",
            help="The title of your movie",
            key="title_input"
        )
        
        genre = st.text_input(
            "Genre *",
            value=st.session_state.get('genre', ''),
            placeholder="e.g., Cyberpunk thriller",
            help="The genre or style of your movie",
            key="genre_input"
        )
    
    with col2:
        num_scenes = st.number_input(
            "Number of Scenes",
            min_value=1,
            max_value=20,
            value=6,
            help="How many scenes to generate"
        )
        
        # Advanced options expander
        with st.expander("🔧 Advanced Options"):
            show_progress = st.checkbox("Show generation progress", value=True)
            include_notes = st.checkbox("Include development notes", value=False)
    
    logline = st.text_area(
        "Logline *",
        value=st.session_state.get('logline', ''),
        placeholder="A rookie hacker and a disillusioned cop team up for a one-night heist in a neon-soaked megacity.",
        help="A one-sentence description of your movie's plot",
        height=80,
        key="logline_input"
    )
    
    # Example presets
    st.markdown("### 💡 Try These Examples")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🌆 Cyberpunk Heist"):
            st.session_state.title = "Neon Heist"
            st.session_state.logline = "A rookie hacker and a disillusioned cop team up for a one-night heist in a neon-soaked megacity."
            st.session_state.genre = "Cyberpunk thriller"
            st.rerun()
    
    with col2:
        if st.button("🚀 Space Adventure"):
            st.session_state.title = "Stellar Exodus"
            st.session_state.logline = "When Earth becomes uninhabitable, a diverse crew must navigate political intrigue and alien encounters to establish humanity's first interstellar colony."
            st.session_state.genre = "Science fiction adventure"
            st.rerun()
    
    with col3:
        if st.button("🏰 Fantasy Quest"):
            st.session_state.title = "The Last Keeper"
            st.session_state.logline = "A young librarian discovers she's the last guardian of ancient magic and must unite fractured kingdoms before an ancient evil awakens."
            st.session_state.genre = "Epic fantasy"
            st.rerun()
    
    # Get current values (either from input fields or session state)
    current_title = title if title else st.session_state.get('title', '')
    current_logline = logline if logline else st.session_state.get('logline', '')
    current_genre = genre if genre else st.session_state.get('genre', '')
    
    st.markdown("---")
    
    # Generate button
    if st.button("🎬 Generate Screenplay", type="primary", use_container_width=True):
        if not title or not logline or not genre:
            st.error("Please fill in all required fields (marked with *).")
            return
        
        # Clear any existing results
        if 'screenplay_result' in st.session_state:
            del st.session_state.screenplay_result
        
        # Show generation progress
        if show_progress:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            stages = [
                "🎭 Director establishing creative vision...",
                "📝 Scene planner structuring narrative...",
                "👥 Character developer creating profiles...",
                "💬 Dialogue writer crafting conversations...",
                "🔍 Continuity editor reviewing consistency...",
                "📄 Formatter preparing final screenplay..."
            ]
        
        try:
            with st.spinner("Generating your screenplay..."):
                if show_progress:
                    for i, stage in enumerate(stages):
                        status_text.text(stage)
                        progress_bar.progress((i + 1) / len(stages))
                        
                        # Simulate processing time for better UX
                        if i == 0:
                            import time
                            time.sleep(1)
                
                # Run the movie creation pipeline with custom parameters
                final_state = run_movie_creation(
                    title=title,
                    logline=logline,
                    genre=genre,
                    num_scenes=num_scenes,
                    model_name=st.session_state.get('selected_model', default_model),
                    temperature=st.session_state.get('selected_temperature', default_temp)
                )
                
                if show_progress:
                    status_text.text("✅ Screenplay generation complete!")
                    progress_bar.progress(1.0)
                
                # Store result in session state
                st.session_state.screenplay_result = final_state
                
        except Exception as e:
            st.error(f"❌ Error generating screenplay: {str(e)}")
            if st.checkbox("Show detailed error"):
                st.code(str(e))
            return
    
    # Display results if available
    if 'screenplay_result' in st.session_state:
        final_state = st.session_state.screenplay_result
        
        st.success("🎉 Screenplay generated successfully!")
        
        # Get formatted content
        formatted_content = final_state.get("formatted_screenplay", {})
        fountain_screenplay = formatted_content.get("fountain_screenplay", "")
        markdown_screenplay = formatted_content.get("markdown_screenplay", "")
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            total_pages = formatted_content.get("total_estimated_pages", "Unknown")
            st.metric("Estimated Pages", total_pages)
        
        with col2:
            character_count = len(formatted_content.get("character_list", []))
            st.metric("Characters", character_count)
        
        with col3:
            st.metric("Scenes", num_scenes)
        
        # Display tabs for different views
        tab1, tab2, tab3 = st.tabs(["📖 Readable Format", "🎬 Fountain Format", "📊 Development Notes"])
        
        with tab1:
            if markdown_screenplay:
                # Display markdown content using Streamlit's native markdown rendering
                # This provides proper markdown structure with responsive formatting
                st.markdown(markdown_screenplay)
            else:
                st.warning("No markdown format available")
        
        with tab2:
            if fountain_screenplay:
                st.code(fountain_screenplay, language="text")
            else:
                st.warning("No fountain format available")
        
        with tab3 if include_notes else st.container():
            if include_notes:
                # Show development notes
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🎭 Director's Vision")
                    director_vision = final_state.get("director_vision", {})
                    if director_vision:
                        st.json(director_vision)
                    
                    st.subheader("👥 Characters")
                    characters = final_state.get("characters", {})
                    if characters:
                        st.json(characters)
                
                with col2:
                    st.subheader("📋 Scene Plan")
                    scene_plan = final_state.get("scene_plan", {})
                    if scene_plan:
                        st.json(scene_plan)
                    
                    st.subheader("🔍 Continuity Review")
                    continuity = final_state.get("continuity_review", {})
                    if continuity:
                        st.json(continuity)
        
        # Download buttons
        st.markdown("### 📥 Download Files")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if fountain_screenplay:
                st.download_button(
                    label="📄 Download Fountain (.fountain)",
                    data=fountain_screenplay,
                    file_name=f"{title.lower().replace(' ', '_')}.fountain",
                    mime="text/plain"
                )
        
        with col2:
            if markdown_screenplay:
                st.download_button(
                    label="📝 Download Markdown (.md)",
                    data=markdown_screenplay,
                    file_name=f"{title.lower().replace(' ', '_')}.md",
                    mime="text/markdown"
                )
        
        with col3:
            # Create ZIP file with both formats
            if fountain_screenplay or markdown_screenplay:
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                    if fountain_screenplay:
                        zip_file.writestr(f"{title.lower().replace(' ', '_')}.fountain", fountain_screenplay)
                    if markdown_screenplay:
                        zip_file.writestr(f"{title.lower().replace(' ', '_')}.md", markdown_screenplay)
                
                zip_buffer.seek(0)
                
                st.download_button(
                    label="📦 Download Both (.zip)",
                    data=zip_buffer.getvalue(),
                    file_name=f"{title.lower().replace(' ', '_')}_screenplay.zip",
                    mime="application/zip"
                )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Multi-Agent Screenplay Generation System | <a href="https://github.com/drMy5tery/MAS" target="_blank">View on GitHub</a></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
