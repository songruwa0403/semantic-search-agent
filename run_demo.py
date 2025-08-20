#!/usr/bin/env python3
"""
🌟 Semantic Search Magic Demo - Launcher
Quick launcher for the semantic search demonstration
"""

import os
import sys
import subprocess
import webbrowser
import time

def main():
    print("🌟 Starting Semantic Search Magic Demo...")
    print("=" * 50)
    
    # Change to project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    
    print(f"📁 Project directory: {project_dir}")
    
    # Check if virtual environment exists
    venv_path = os.path.join(project_dir, "venv")
    if not os.path.exists(venv_path):
        print("❌ Virtual environment not found!")
        print("Please ensure you're in the correct project directory.")
        return
    
    # Start the demo
    print("🚀 Launching Semantic Search Magic Demo...")
    print()
    print("🧠 This will demonstrate:")
    print("   • AI-powered semantic search vs traditional keyword search")
    print("   • Real fitness discussion data (1000+ posts)")
    print("   • Side-by-side comparison with explanations")
    print("   • Confidence indicators and similarity scores")
    print()
    print("✨ Try these magic searches:")
    print("   • 'knee hurt when exercising'")
    print("   • 'struggling to lose weight'")
    print("   • 'beginner workout anxiety'")
    print("   • 'muscle building plateau'")
    print()
    print("🔄 Starting web server...")
    
    try:
        # Activate virtual environment and run the app
        if sys.platform == "win32":
            activate_script = os.path.join(venv_path, "Scripts", "activate.bat")
            python_exe = os.path.join(venv_path, "Scripts", "python.exe")
        else:
            activate_script = os.path.join(venv_path, "bin", "activate")
            python_exe = os.path.join(venv_path, "bin", "python")
        
        # Change to frontend directory
        frontend_dir = os.path.join(project_dir, "stage-4-interface", "frontend")
        os.chdir(frontend_dir)
        
        # Start the Flask app
        app_process = subprocess.Popen([
            python_exe, "app.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Wait a moment for the server to start
        print("⏳ Initializing AI models (this may take 30-60 seconds)...")
        time.sleep(5)
        
        # Try to open browser
        url = "http://localhost:5001"
        print(f"🌐 Opening browser to: {url}")
        try:
            webbrowser.open(url)
        except:
            print("Could not auto-open browser. Please manually navigate to:")
            print(f"   {url}")
        
        print()
        print("🎉 Demo is running!")
        print("📱 The web interface should open automatically")
        print("⌨️  Press Ctrl+C to stop the demo")
        print()
        
        # Wait for user to stop
        try:
            app_process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping demo...")
            app_process.terminate()
            app_process.wait()
            print("✅ Demo stopped successfully!")
        
    except Exception as e:
        print(f"❌ Error starting demo: {e}")
        print("Please try running manually:")
        print("   cd stage-4-interface/frontend")
        print("   source ../../venv/bin/activate")
        print("   python app.py")

if __name__ == "__main__":
    main()
