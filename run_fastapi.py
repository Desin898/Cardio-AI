#!/usr/bin/env python3
"""
Coronary AI Detection System - FastAPI Server Launch Script
Run this file to start the enterprise FastAPI application
"""
import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    print("=" * 60)
    print("🫀 CORONARY AI DETECTION SYSTEM - FASTAPI ENTERPRISE SERVER")
    print("=" * 60)
    print(f"🌐 Server running at: http://localhost:{port}")
    print(f"📚 OpenAPI Documentation: http://localhost:{port}/docs")
    print(f"📖 ReDoc Documentation: http://localhost:{port}/redoc")
    print("=" * 60)
    
    uvicorn.run("backend.main:app", host=host, port=port, reload=True)
