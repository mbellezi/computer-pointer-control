from argparse import ArgumentParser

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-t", "--input-type", required=True, type=str,
                        help="Type of input (video, image or cam)")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Input file")
    parser.add_argument("-o", "--out", type=str, default=None,
                        help="Output file with the processed content")

    return parser


def main():
    args = build_argparser().parse_args()

if __name__ == '__main__':
    main()