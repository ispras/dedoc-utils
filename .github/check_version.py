import argparse
import re
from typing import Pattern


def is_correct_version(version: str, tag: str, old_version: str, regexp: Pattern) -> bool:
    match = regexp.match(version)

    if match is None:
        print("New version doesn't match the pattern")  # noqa
        return False

    if not (tag.startswith("v") and tag[1:] == version):
        print("Tag value should be equal to version with `v` in the beginning")  # noqa
        return False

    return old_version < version


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", help="Tag of the release", type=str)
    parser.add_argument("--new_version", help="New release version", type=str)
    parser.add_argument("--old_version", help="Previous release version", type=str)
    args = parser.parse_args()

    print(f"Old version: {args.old_version}, new version: {args.new_version}, tag: {args.tag}")  # noqa

    version_pattern = re.compile(r"^\d+\.\d+(\.\d+)?$")
    correct = is_correct_version(args.new_version, args.tag, args.old_version, version_pattern)

    assert correct
    print("Version is correct")  # noqa
