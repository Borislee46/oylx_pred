from src.pages.algorithm_lab.pymoo.core.mutation import Mutation


class NoMutation(Mutation):
    def do(self, problem, pop, **kwargs):
        return pop
