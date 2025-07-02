"""
Student Performance Predictor - Main Entry Point
Runs the FastAPI server for the student performance prediction application.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(
        "src.api.BestModelApi:app", 
        host="0.0.0.0",  # Changed from 127.0.0.1 to 0.0.0.0 for Render
        port=port,
        reload=False,  # Disabled reload for production
        log_level="info"
    )