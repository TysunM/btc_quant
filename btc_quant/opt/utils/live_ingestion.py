import asyncio
import zmq
import json
import time
from alpaca.data.live.crypto import CryptoDataStream
import torch

class HetznerPublisher:
    def __init__(self, hetzner_ip, port="5555"):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        
        target = f"tcp://{hetzner_ip}:{port}"
        self.socket.connect(target)
        
        time.sleep(1) 
        print(f"ZMQ Publisher initialized. Connected to {target}")

    def send_signal(self, symbol, action, confidence):
        payload = {
            "symbol": symbol,
            "action": action,
            "confidence": round(confidence, 4)
        }
        self.socket.send_string(json.dumps(payload))
        print(f"Signal Transmitted: {payload}")

    def send_eof(self):
        payload = {
            "symbol": "SYSTEM",
            "action": "EOF",
            "confidence": 1.0
        }
        self.socket.send_string(json.dumps(payload))
        print("EOF Sentinel Transmitted.")
        time.sleep(1)
        self.socket.close()
        self.context.term()

class AlchemicalDataEngine:
    def __init__(self, api_key, api_secret, hetzner_ip):
        self.stream = CryptoDataStream(api_key, api_secret)
        self.publisher = HetznerPublisher(hetzner_ip)
        self.tick_buffer = []
        self.sequence_length = 10 

    async def handle_trade(self, trade):
        self.tick_buffer.append(trade.price)
        print(f"Incoming Tick: {trade.symbol} @ ${trade.price}")

        if len(self.tick_buffer) > self.sequence_length:
            self.tick_buffer.pop(0)

        if len(self.tick_buffer) == self.sequence_length:
            self.run_inference()

    def run_inference(self):
        input_tensor = torch.tensor(self.tick_buffer, dtype=torch.float32).cuda()
        
        current_price = self.tick_buffer[-1]
        avg_price = sum(self.tick_buffer) / len(self.tick_buffer)
        
        action = "BUY" if current_price > avg_price else "SELL"
        confidence = 0.99 

        self.publisher.send_signal(symbol="BTC/USD", action=action, confidence=confidence)
        self.tick_buffer.clear() 

    def start_stream(self):
        print("Igniting Live WebSocket Stream for BTC/USD...")
        self.stream.subscribe_trades(self.handle_trade, "BTC/USD")
        
        try:
            self.stream.run()
        except KeyboardInterrupt:
            print("\nManual intervention detected. Initiating shutdown...")
        finally:
            self.publisher.send_eof()

if __name__ == "__main__":
    # !!! CHANGE THESE IN YOUR TEXT EDITOR AFTER RUNNING THIS BLOCK !!!
    API_KEY = "PK74ZH3L5Q6ODIN7YR4EJQFIWV"
    API_SECRET = "FZtBzjZZrB8WM8u2NjHuGDvgYdBijwBCDE41JS2pnFvY"
    HETZNER_IP = "204.168.222.87" 
    
    engine = AlchemicalDataEngine(API_KEY, API_SECRET, HETZNER_IP)
    engine.start_stream()
