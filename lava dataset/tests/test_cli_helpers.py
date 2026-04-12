import unittest
from argparse import Namespace

from lava_search.cli import parse_weights, resolve_profile


class CliHelperTests(unittest.TestCase):
    def test_parse_weights(self) -> None:
        weights = parse_weights("sparse=0.1,dense=0.4,clip_text=0.2,clip_image=0.3")
        self.assertEqual(
            weights,
            {
                "sparse": 0.1,
                "dense": 0.4,
                "clip_text": 0.2,
                "clip_image": 0.3,
            },
        )

    def test_resolve_profile_with_flags(self) -> None:
        profile = resolve_profile(
            "strongest",
            Namespace(no_sparse=False, no_dense=True, no_clip=False),
        )
        self.assertEqual(profile, (True, False, True))


if __name__ == "__main__":
    unittest.main()
