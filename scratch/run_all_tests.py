import unittest, sys, io, contextlib, pathlib

# Capture output of unittest discovery
loader = unittest.TestLoader()
stream = io.StringIO()
runner = unittest.TextTestRunner(stream=stream, verbosity=2)
suite = loader.discover('tests')
result = runner.run(suite)
print(stream.getvalue())
print('TOTAL=', result.testsRun, 'FAILURES=', len(result.failures), 'ERRORS=', len(result.errors))
