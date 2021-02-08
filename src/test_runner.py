import unittest

class TestCases(unittest.TestCase):
    def test_PRE1(self):
        '''This test will ensure that data is correctly preprocessed and formatted for lift detection'''
        # TODO
        pass

    def test_PRE2(self):
        '''This test will ensure that data is correctly preprocessed and formatted for lift classification'''
        from processing import preprocess
        samples, _, max_size, _ = preprocess('./source', './metadata/lift_times_untrimmed.csv')

        for sample in samples:
            self.assertEqual(sample.shape[0], max_size)
            self.assertEqual(sample.shape[1], 36)

    def test_TRA1(self):
        '''This test will ensure that cross-validation process is followed for each model'''
        # TODO
        pass

    def test_TRA2(self):
        '''This test will ensure that each model of the cross-validation training process is succesfully trained'''
        # TODO
        pass

    def test_DET1(self):
        '''This test will ensure that the lift detection model can detect a lift is occurring with a high degree of confidence'''
        # TODO
        pass

    def test_DET2(self):
        '''This test will ensure that the lift detection model can detect a lift is *not* occurring with a high degree of confidence'''
        # TODO
        pass

    def test_DET3(self):
        '''This test will ensure that if the lift detection model detects a lift, the lift classification process starts'''
        # TODO
        pass

    def test_DET4(self):
        '''This test will ensure that if the lift detection model does not detect a lift, the lift classification progress does not start'''
        # TODO
        pass

    def test_CLA1(self):
        '''This test will ensure that the lift classification model can classify a low-risk lift with a high degree of confidence'''
        # TODO
        pass

    def test_CLA2(self):
        '''This test will ensure that the lift classification model can classify a medium-risk lift with a high degree of confidence'''
        # TODO
        pass

    def test_CLA3(self):
        '''This test will ensure that the lift classification model can classify a high-risk lift with a high degree of confidence'''
        # TODO
        pass

    def test_CLA4(self):
        '''This test will ensure that the lift classification model can classify a sample that is not a lift with a high degree of confidence'''
        # TODO
        pass

    def test_END1(self):
        '''This test will ensure that the entire process returns negative if a lift is not present'''
        # TODO
        pass

    def test_END2(self):
        '''This test will ensure that the entire process returns low-risk if a low-risk lift is detected'''
        # TODO
        pass

    def test_END3(self):
        '''This test will ensure that the entire process returns medium-risk if a medium-risk lift is detected'''
        # TODO
        pass

    def test_END4(self):
        '''This test will ensure that the entire process returns high-risk if a high-risk lift is detected'''
        # TODO
        pass

    def test_END5(self):
        '''This test will ensure that the entire process returns low-risk if a low-risk lift is detected'''
        # TODO
        pass

    def test_END6(self):
        '''This test will ensure that the entire process returns negative if a lift is not present but is not flagged by the detection model'''
        # TODO
        pass

    def test_END7(self):
        '''This test will ensure that the entire process correctly classifies lift models'''
        # TODO
        pass

    def test_ABN1(self):
        '''This test will determine the detection model's performance on unknown/abnormal data sequences'''
        # TODO
        pass

    def test_ABN2(self):
        '''This test will determine the classification model's performance on unknown/abnormal data sequences'''
        # TODO
        pass

if __name__ == '__main__':
    unittest.main()