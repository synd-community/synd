import pytest


def test_invoke_cli_app() -> None:
    with pytest.raises(ValueError):
        raise ValueError("Halt and catch fire")


if __name__ == "__main__":
    pytest.main(["-x", __file__, "--no-cov", "-vv"])
