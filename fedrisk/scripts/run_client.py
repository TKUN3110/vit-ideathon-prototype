import argparse
import flwr as fl
import time
from fedrisk.federated.client import FedRiskClient

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site_id", type=int, required=True)
    parser.add_argument("--server", type=str, default="127.0.0.1:8080")
    args = parser.parse_args()

    print(f"[Client {args.site_id}] Waiting for server to start...")
    time.sleep(10)
    
    print(f"[Client {args.site_id}] Starting Flower client, connecting to {args.server}...")
    client = FedRiskClient(site_id=args.site_id)
    fl.client.start_client(server_address=args.server, client=client.to_client())

if __name__ == "__main__":
    main()
