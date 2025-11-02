#!/usr/bin/env python3
"""Simple WebSocket server for real-time SOHO updates."""
import asyncio
import websockets
import json
from datetime import datetime

# Simulated detection stream (replace with your pipeline hooks)
async def generate_detections(websocket, path):
    """Simulate streaming new candidates (integrate with detect_comets.py)."""
    clients = set()
    clients.add(websocket)

    try:
        # Example: Stream fake detections every 10s
        for i in range(5):  # Or loop forever
            fake_detection = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "id": f"C2#{i}",
                "label": "comet" if i % 2 == 0 else "not_comet",
                "score": round(0.8 + 0.1 * i, 2),
                "position": [400 + i * 10, 500 + i * 5]
            }
            msg = json.dumps({"type": "new_detection", "data": fake_detection})
            for client in list(clients):
                try:
                    await client.send(msg)
                except websockets.exceptions.ConnectionClosed:
                    clients.remove(client)
            await asyncio.sleep(10)  # Simulate pipeline delay
    except websockets.exceptions.ConnectionClosed:
        pass

# Start server
start_server = websockets.serve(generate_detections, "localhost", 8765)

print("WebSocket server started on ws://localhost:8765")
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
