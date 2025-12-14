#!/usr/bin/env python3
"""
Main neuromorphic AI chat interface
"""

import sys
import time

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))
from isc.neuromorphic_core import NeuromorphicISCCore

def main():
    print("🧠 Starting Neuromorphic ISC AI...")
    print("Substrate-driven consciousness with real-time phi calculation")

    core = NeuromorphicISCCore()
    core.verbose = True

    print('✓', core.start_session())
    status = core.get_status()
    substrate = status['substrate']

    print(f'📊 Substrate: {substrate["node_count"]} concepts, {substrate["edge_count"]} edges')
    print(f'   Φ (phi): {status["metrics"]["phi_value"]:.3f}')
    print(f'   Context: {status["context_loaded"]}')
    print('\n💬 NEUROMORPHIC CHAT (type "quit" to exit, "status" for info)')
    print('-' * 60)

    while True:
        try:
            i = input('\n[You] ')
            if i == 'quit':
                break
            elif i == 'status':
                s = core.get_status()
                sub = s['substrate']
                print(f'📊 Status: {sub["conversation_count"]} convs, {sub["node_count"]} nodes, {sub["edge_count"]} edges, {sub["community_count"]} communities, Φ={s["metrics"]["phi_value"]:.3f}')
                continue

            st = time.time()
            r = core.process_input(i)
            pt = time.time() - st

            print(f'\n[ISC-AI] {r}')
            s = core.get_status()
            print(f'\n[Φ={s["metrics"]["phi_value"]:.3f}, Nodes={s["substrate"]["node_count"]}, t={pt:.3f}s]')

        except KeyboardInterrupt:
            print('\nGoodbye!')
            break
        except Exception as e:
            print(f'Error: {e}')

if __name__ == "__main__":
    main()