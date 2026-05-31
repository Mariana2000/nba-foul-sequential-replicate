import argparse

from whistle_balance.validation import main as validate_main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare parsed fouls to box score totals.")
    parser.add_argument("--sample-size", type=int, default=10)
    args = parser.parse_args()
    validate_main(sample_size=args.sample_size)
