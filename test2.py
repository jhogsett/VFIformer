import argparse

parser = argparse.ArgumentParser(description='resequence video frames')
parser.add_argument("--verbose", dest="verbose", default=False, action="store_true", help="show extra details")
args = parser.parse_args()
print(args)
