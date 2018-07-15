import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--faces_path", help="Path to the directory of faces")
parser.add_argument("--encodings_file", help="Where to save the encodings file")
args = parser.parse_args()

if args.faces_path is None:
    print("A faces directory path must be passed with --faces_path <faces directory>")
    sys.exit()

if args.encodings_file is None:
    print("A path for the encodings file must be passed with --encodings_file encodings.p")
    sys.exit()

from personable import Tracker

tracker = Tracker()
tracker.create_encodings(args.faces_path)
tracker.save_encodings(args.encodings_file)