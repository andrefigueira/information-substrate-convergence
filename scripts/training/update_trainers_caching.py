#!/usr/bin/env python3
"""
Script to update all trainer scripts to use caching
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Template for adding caching to trainers
CACHING_IMPORT = """from isc.cache_manager import CacheManager"""

CACHING_INIT = """        # Initialize cache manager
        self.cache_manager = CacheManager(cache_dir="trainer_cache", max_memory_items=1000)"""

CACHING_METHOD = '''
    def _get_chatgpt_response(self, messages: List[Dict[str, str]], model: str = "gpt-3.5-turbo", temperature: float = 0.7, max_tokens: int = 150) -> Tuple[str, bool]:
        """Get ChatGPT response with caching"""
        # Create cache key from messages
        prompt = json.dumps(messages)
        
        # Check cache
        cached_response = self.cache_manager.get_chatgpt_response(prompt, model=model)
        if cached_response:
            self.session_metrics["cache_hits"] = self.session_metrics.get("cache_hits", 0) + 1
            return cached_response, True
        
        self.session_metrics["cache_misses"] = self.session_metrics.get("cache_misses", 0) + 1
        
        # Get new response
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=30
            )
            
            content = response.choices[0].message.content.strip()
            
            # Cache the response
            self.cache_manager.save_chatgpt_response(prompt, content, model=model)
            
            # Track usage
            if hasattr(response, 'usage'):
                self.session_metrics["tokens_used"]["prompt"] += response.usage.prompt_tokens
                self.session_metrics["tokens_used"]["completion"] += response.usage.completion_tokens
            
            return content, False
            
        except Exception as e:
            raise Exception(f"ChatGPT API error: {e}")
'''

def update_trainer_file(filepath: str):
    """Update a trainer file to use caching"""
    print(f"Updating {filepath}...")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Check if already has caching
    if 'CacheManager' in content:
        print(f"  - Already has caching, skipping")
        return
    
    # Add import
    import_pos = content.find('from isc.core import ISCCore')
    if import_pos != -1:
        import_end = content.find('\n', import_pos)
        content = content[:import_end] + '\n' + CACHING_IMPORT + content[import_end:]
    
    # Add to __init__ method
    init_pos = content.find('def __init__(')
    if init_pos != -1:
        # Find the end of __init__ assignments
        init_end = content.find('def ', init_pos + 1)
        if init_end != -1:
            # Insert before the next method
            content = content[:init_end] + CACHING_INIT + '\n\n' + content[init_end:]
    
    # Add the caching method
    class_end = content.rfind('class ')
    next_class = content.find('class ', class_end + 1)
    if next_class == -1:
        # No next class, add before main
        main_pos = content.find('def main(')
        if main_pos != -1:
            content = content[:main_pos] + CACHING_METHOD + '\n\n' + content[main_pos:]
    else:
        content = content[:next_class] + CACHING_METHOD + '\n\n' + content[next_class:]
    
    # Update metrics initialization
    metrics_pos = content.find('self.session_metrics = {')
    if metrics_pos != -1:
        metrics_end = content.find('}', metrics_pos)
        if metrics_end != -1 and '"cache_hits"' not in content[metrics_pos:metrics_end]:
            # Add cache metrics
            content = content[:metrics_end] + '            "cache_hits": 0,\n            "cache_misses": 0,\n' + content[metrics_end:]
    
    # Save updated file
    backup_path = filepath + '.backup'
    os.rename(filepath, backup_path)
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"  - Updated successfully (backup saved as {backup_path})")

def main():
    """Update all trainer files"""
    scripts_dir = Path(__file__).parent
    
    trainers = [
        'chatgpt_trainer.py',
        'chatgpt_trainer_enhanced.py',
        'chatgpt_trainer_multithreaded.py'
    ]
    
    print("Updating trainer scripts to use caching...\n")
    
    for trainer in trainers:
        filepath = scripts_dir / trainer
        if filepath.exists():
            try:
                update_trainer_file(str(filepath))
            except Exception as e:
                print(f"  - Error updating {trainer}: {e}")
        else:
            print(f"  - {trainer} not found")
    
    print("\n✓ Update complete!")
    print("\nNote: The trainers now need to update their ChatGPT API calls to use the _get_chatgpt_response method.")
    print("You may need to manually update the specific API calls in each trainer.")

if __name__ == "__main__":
    main()