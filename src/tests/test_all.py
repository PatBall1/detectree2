import unittest

from src.tests import test_data
from src.tests import test_preprocessing
from src.tests import test_models

suites = []
suites.append(test_data.suite)
suites.append(test_preprocessing.suite)
suites.append(test_models.suite)

suite = unittest.TestSuite(suites)

if __name__ == "__main__":  # pragma: no cover
    unittest.TextTestRunner(verbosity=2).run(suite)
