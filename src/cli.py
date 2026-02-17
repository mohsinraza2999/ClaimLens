import argparse
import sys, os
import uvicorn
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_pipeline.preprocessing import main
from src.training.tune_train import tp_main
from src.inference.main import app


def cli_main():
    parser = argparse.ArgumentParser(description="ClaimLens")
    parser.add_argument("command",choices=["process", "train", "route"],
                        help="choose from the following: process, train, and route")
    args=parser.parse_args()
    if args.command=="process":
        main()
    elif args.command=="train":
        tp_main()
    elif args.command=="route":
        uvicorn.run(app,host="0.0.0.0", port=8000)

if __name__=="__main__":
    cli_main()