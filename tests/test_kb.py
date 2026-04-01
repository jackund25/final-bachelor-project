# test_kb.py
from src.rag.knowledge_base import MedicalKnowledgeBase

kb = MedicalKnowledgeBase()
kb.create_manual_kb()  # Use built-in manual KB
print(f"✓ Created {len(kb.chunks)} knowledge chunks")