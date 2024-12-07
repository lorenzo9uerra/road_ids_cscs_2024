import argparse
import sys


def get_parser():
    parser = argparse.ArgumentParser(description="LSTM CAN Bus Anomaly Detection")
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        choices=[
            "road",
            "car-hacking",
            "in-vehicle_chevrolet",
            "in-vehicle_hyundai",
            "in-vehicle_kia",
        ],
        help="Dataset to use",
    )
    parser.add_argument(
        "-l",
        "--label",
        type=int,
        required=False,
        choices=list(range(1, 12)),
        help="Label to use for binary classification in road dataset",
    )
    parser.add_argument(
        "-ct",
        "--classification_type",
        type=str,
        choices=["multiclass", "binary"],
        required=True,
        help="Type of attack classification",
    )
    parser.add_argument(
        "--retrain", action="store_true", help="If true, retrain the model"
    )
    parser.add_argument(
        "--test", action="store_true", help="If true, only test the model"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["LSTM", "DCNN"],
        required=True,
        help="Model to use",
    )
    args = parser.parse_args()

    if args.model == "DCNN" and args.classification_type != "binary":
        sys.exit("error: DCNN model only supports binary classification")
    if (
        args.dataset == "road"
        and args.classification_type == "binary"
        and args.label is None
    ):
        sys.exit("error: label is required for binary classification in road dataset")
    return args
