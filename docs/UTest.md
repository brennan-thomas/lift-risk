# Testing Framework

Testing functionality is provided by the `unittest` package in the Python Standard Library and is implemented in [test_runner.py](../src/test_runner.py). Tests can be run individually or as a full suite. 

For the full test suite, navigate to the [src](../src) directory and run

`python test_runner.py`.

This will test the full functionality of the repository. The `-v` flag can be added for more verbose output.

To run an individual test, run the command

`python -m unittest test_runner.TestCases.test_[ID]`

where `[ID]` is the test ID as defined in [Test_Plan.md](../Test_Plan.md). For example,

`python -m unittest test_runner.TestCases.test_PRE2`

will run the test "Preprocessing Test 2".