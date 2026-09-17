"""Console entry point for the crystallography-constrained growth engine."""
from .growth import parse_args, run


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
