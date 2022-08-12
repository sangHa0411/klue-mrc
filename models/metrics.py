
from datasets import load_metric

class SquadMetric:
    def __init__(self,):
        self.metric_v2 = load_metric("squad_v2")

    def compute_metrics(self, predictinos, labels):
        results = self.metric_v2.compute(
            predictions=predictinos, references=labels, no_answer_threshold=0.0
        )
        return results