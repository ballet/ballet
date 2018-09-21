class ProblemType:
    classification = False
    binary_classification = False
    multi_classification = False
    regression = False


class ClassificationProblem(ProblemType):
    classification = True


class BinaryClassificationProblem(ClassificationProblem):
    binary_classification = True


class MulticlassClassificationProblem(ClassificationProblem):
    multi_classification = True


class RegressionProblem(ProblemType):
    regression = True
