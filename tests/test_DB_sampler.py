import pytest
#import dbopt.DB_sampler
from dbopt.DB_sampler import DB_sampler
from dbopt.skeleton import fib, main

__author__ = "Noam Ghenassia"
__copyright__ = "Noam Ghenassia"
__license__ = "MIT"


def test_fib():
    """API Tests"""
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)


def test_main(capsys):
    """CLI Tests"""
    # capsys is a pytest fixture that allows asserts agains stdout/stderr
    # https://docs.pytest.org/en/stable/capture.html
    main(["7"])
    captured = capsys.readouterr()
    assert "The 7-th Fibonacci number is 13" in captured.out

def test_DB_sampler():
    x = dbopt.DB_sampler.DB_sampler(5, 15)#instancier objet ici
    assert x.key == 15#tester le comportement de l'objet ici