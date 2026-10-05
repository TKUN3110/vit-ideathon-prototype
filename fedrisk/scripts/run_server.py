import argparse
import flwr as fl
from fedrisk.federated.strategy import FedRiskSecAggStrategy
from fedrisk.config import NUM_FEDERATED_ROUNDS
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=NUM_FEDERATED_ROUNDS)
    parser.add_argument("--host", type=str, default="0.0.0.0:8080")
    args = parser.parse_args()

    print(f"[Server] Starting Flower aggregator on {args.host} for {args.rounds} rounds...")

    def state_callback(round_data):
        print(f"[Server] Round {round_data.get('round')} complete: val_loss={round_data.get('val_loss')} val_auroc={round_data.get('val_auroc')}")
        # Log to disk for backend/frontend to read
        log_path = Path("logs/federated_history.json")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        history = []
        if log_path.exists():
            with open(log_path, "r") as f:
                history = json.load(f)
        history.append(round_data)
        with open(log_path, "w") as f:
            json.dump(history, f, indent=2)

    strategy = FedRiskSecAggStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        smpc_enabled=True,
        state_callback=state_callback
    )

    fl.server.start_server(
        server_address=args.host,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
