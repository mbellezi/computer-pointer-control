import cv2
from argparse import ArgumentParser
from input_feeder import InputFeeder


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


def initialize_feed(input_type, input):
    feed = InputFeeder(input_type, input)
    feed.load_data()
    return feed


def initialize_window():
    cv2.namedWindow('preview')


def infer_frame(frame, feed_type):
    cv2.imshow('preview', frame)
    key_pressed = cv2.waitKey(1)
    if key_pressed == 27:
        return False


def process_feed(feed, feed_type):
    for batch in feed.next_batch():
        if batch is not False:
            if infer_frame(batch, feed_type) is False:
                return
        else:
            return


def main():
    args = build_argparser().parse_args()
    feed = initialize_feed(args.input_type, args.input)
    initialize_window()
    process_feed(feed, args.input_type)
    feed.close()


if __name__ == '__main__':
    main()
