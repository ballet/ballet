class Problem:

    def is_classification(self):
        return False

    def is_binary_classification(self):
        return False

    def is_multi_classification(self):
        return False

    def is_regression(self):
        return False


class ClassificationProblem(Problem):
    def is_classification(self):
        return True


class BinaryClassificationProblem(ClassificationProblem):
    def is_binary_classification(self):
        return True


class MulticlassClassificationProblem(ClassificationProblem):
    def is_multi_classification(self):
        return True


class RegressionProblem(Problem):
    def is_regression(self):
        return True


class ProblemTypes:
    CLASSIFICATION = ClassificationProblem()
    REGRESSION = RegressionProblem()
    BINARY_CLASSIFICATION = BinaryClassificationProblem()
    MULTI_CLASSIFICATION = MulticlassClassificationProblem()
