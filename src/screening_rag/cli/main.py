import argparse

from screening_rag.cli import initialize, renew, report
from screening_rag.custom_types import SortingBy


def main():
    parser = argparse.ArgumentParser()
    subcmd = parser.add_subparsers(dest="command", required=True)

    # init
    init_parser = subcmd.add_parser("init")
    init_parser.add_argument(
        "--keywords",
        help="The keywords to search on CNN",
        type=str,
    )
    init_parser.add_argument(
        "--amount", help="The amount of the crawled articles", type=int
    )
    init_parser.add_argument(
        "-s",
        "--sortby",
        help="The factor of news ranking",
        default=SortingBy.RELEVANCY,
    )
    init_parser.set_defaults(func=initialize.initialize_system)

    # renew
    renew_parser = subcmd.add_parser("renew")

    renew_parser.add_argument(
        "--keywords",
        help="The keywords to search on CNN",
        type=str,
    )
    renew_parser.add_argument(
        "-s",
        "--sortby",
        help="The factor of news ranking",
        default=SortingBy.NEWEST,
    )
    renew_parser.set_defaults(func=renew.renew_system)

    # report
    report_parser = subcmd.add_parser("report")
    report_parser.set_defaults(func=report.main)

    args = parser.parse_args()

    if args.command == "init":
        args.func(args.keywords, args.amount, args.sortby)
    elif args.command == "renew":
        args.func(args.keywords, args.sortby)
    elif args.command == "report":
        args.func()
    else:
        raise ValueError("Please use init, renew or report")
