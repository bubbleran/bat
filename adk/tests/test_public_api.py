import bat


def test_everything_is_reachable_from_bat():
    for name in bat.__all__:
        assert getattr(bat, name) is not None
