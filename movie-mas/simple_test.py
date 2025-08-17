#!/usr/bin/env python3
"""
Simple test to check if agents work without LangGraph
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_basic_groq():
    """Test basic Groq connection"""
    try:
        from langchain_groq.chat_models import ChatGroq
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import JsonOutputParser
        
        print("✅ All required imports successful")
        
        # Test basic Groq connection
        llm = ChatGroq(
            model_name=os.getenv("GROQ_MODEL_NAME", "llama3-8b-8192"),
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.7
        )
        
        # Simple test prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Respond with a JSON object containing only: {\"message\": \"Hello from Groq!\", \"status\": \"success\"}"),
            ("user", "Say hello")
        ])
        
        parser = JsonOutputParser()
        chain = prompt | llm | parser
        
        result = chain.invoke({"input": "test"})
        print(f"✅ Groq connection successful: {result}")
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Run: pip install langchain-groq langchain-core")
        return False
    except Exception as e:
        print(f"❌ Groq connection failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Simple Agent Test")
    print("=" * 30)
    
    if not os.getenv("GROQ_API_KEY"):
        print("❌ GROQ_API_KEY not found")
        exit(1)
    
    print(f"Model: {os.getenv('GROQ_MODEL_NAME', 'llama3-8b-8192')}")
    
    if test_basic_groq():
        print("\n✅ Basic test passed - your setup is working!")
        print("Now install remaining dependencies: pip install -r requirements.txt")
    else:
        print("\n❌ Basic test failed - check your setup")
