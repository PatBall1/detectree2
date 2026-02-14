"""detectree2: Automatic tree crown delineation using Mask R-CNN.

Requires `Detectron2 <https://github.com/facebookresearch/detectron2>`_
and `PyTorch <https://pytorch.org>`_ to be installed separately.

Install PyTorch first (see https://pytorch.org/get-started), then::

    pip install 'git+https://github.com/facebookresearch/detectron2.git'
    pip install detectree2
"""

__version__ = "2.1.2"


def _check_detectron2() -> None:
    """Verify that detectron2 is importable and give a helpful error if not."""
    try:
        import detectron2  # noqa: F401
    except ImportError:
        raise ImportError(
            "\n"
            "detectree2 requires Facebook's Detectron2, which must be installed separately.\n"
            "\n"
            "  1. Install PyTorch first:  https://pytorch.org/get-started\n"
            "  2. Then install Detectron2:\n"
            "       pip install 'git+https://github.com/facebookresearch/detectron2.git'\n"
            "\n"
            "See: https://detectron2.readthedocs.io/en/latest/tutorials/install.html\n"
        ) from None


_check_detectron2()
