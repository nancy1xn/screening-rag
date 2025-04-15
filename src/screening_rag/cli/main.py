from argparse import ArgumentParser

from screening_rag.cli.initialize import initialize_system
from screening_rag.cli.renew import renew_system
from screening_rag.custom_types import SortingBy

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["mode_initialize", "mode_renew"],
        required=True,
        help="choose mode_initialize or mode_renew",
    )
    parser.add_argument("--keyword", help="The keyword to search on CNN", type=str)
    parser.add_argument("--amount", help="The amount of the crawled articles", type=int)
    parser.add_argument(
        "-s",
        "--sort-by",
        help="The factor of news ranking",
        default=SortingBy.RELEVANCY,
    )
    args = parser.parse_args()

    if args.mode == "mode_initialize":
        keywords = ["JP Morgan financial crime"]
        initialize_system(keywords, args.amount, SortingBy.RELEVANCY)

    if args.mode == "mode_renew":
        keywords = ["JP Morgan financial crime"]
        renew_system(keywords, SortingBy.NEWEST)
