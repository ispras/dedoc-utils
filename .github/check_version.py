import argparse
import re

from pkg_resources import parse_version

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", help="Tag of the release", type=str)
    parser.add_argument("--new_version", help="New release version", type=str)
    parser.add_argument("--old_version", help="Previous release version", type=str)
    args = parser.parse_args()

    print(f"Old version: {args.old_version}, new version: {args.new_version}, tag: {args.tag}")  # noqa

    version_pattern = re.compile(r"^\d+\.\d+(\.\d+)?$")
    match = version_pattern.match(args.new_version)

    assert match is not None, "New version doesn't match the pattern"
    assert args.tag.startswith("v") and args.tag[1:] == args.new_version, "Tag value should be equal to version with `v` in the beginning"
    assert parse_version(args.old_version) < parse_version(args.new_version), "New version should be greater than old version"

    print("Version is correct")  # noqa
