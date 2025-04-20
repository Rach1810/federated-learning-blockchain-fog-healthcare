#!/bin/bash

echo "🔄 Creating virtual environment..."
python3.10 -m venv venv
source venv/bin/activate

echo "📦 Installing dependencies..."
pip install -r requirements.txt

echo "🚀 Running simulation..."
python simulate.py
