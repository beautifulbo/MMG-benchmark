"""Test script to verify CRLMMNAR model registration."""
import sys
import os

# Add MMG-benchmark to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing CRLMMNAR model registration...")
print("=" * 50)

try:
    from models.registry import ModelRegistry
    
    # List all registered models
    registered = ModelRegistry.list_models()
    print(f"Registered models: {registered}")
    
    # Try to get CRLMMNAR
    print("\nAttempting to load CRLMMNAR...")
    crlmmnar = ModelRegistry.get('CRLMMNAR')
    
    if crlmmnar is not None:
        print("✓ SUCCESS: CRLMMNAR model loaded!")
        print(f"  Class: {crlmmnar.__name__}")
        print(f"  Module: {crlmmnar.__module__}")
        
        # Check if it's a proper subclass
        from models.base.abstract_recommender import GeneralRecommender
        if issubclass(crlmmnar, GeneralRecommender):
            print("  ✓ Correctly inherits from GeneralRecommender")
        else:
            print("  ✗ Does NOT inherit from GeneralRecommender")
            
        print("\n" + "=" * 50)
        print("Model registration VERIFIED successfully!")
    else:
        print("✗ FAILED: CRLMMNAR model not found in registry")
        
except Exception as e:
    print(f"✗ ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
