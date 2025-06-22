#!/usr/bin/env python3
"""
Simple script to start the API for testing
"""
import uvicorn
import sys
import os

if __name__ == "__main__":
    print("Starting RSVP Forecast API...")
    print("API will be available at: http://localhost:8000")
    print("API docs will be available at: http://localhost:8000/docs")
    print("Press Ctrl+C to stop")
    
    try:
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    except KeyboardInterrupt:
        print("\nAPI stopped.")
    except Exception as e:
        print(f"Error starting API: {e}")
        sys.exit(1)
