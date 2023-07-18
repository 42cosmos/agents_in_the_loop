import itertools
import logging
import numpy as np

import evaluate
from seqeval.metrics import classification_report


class Metrics:
    def __init__(self, args, label_list, do_report=True):
        self.args = args
        self.label_list = label_list
        self.id_to_label = dict(enumerate(label_list))
        self.metric = evaluate.load("seqeval")
        self.report = do_report
        self.logger = logging.getLogger(__name__)

    def compute_metrics(self, p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [self.id_to_label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.id_to_label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = self.metric.compute(predictions=true_predictions,
                                      references=true_labels,
                                      zero_division=0)

        if self.report:
            try:
                self.logger.info(f"\n {classification_report(true_labels, true_predictions, suffix=False)}")
            except ValueError as v:
                labels = set(list(itertools.chain(*true_labels)))
                pred = set(list(itertools.chain(*true_predictions)))
                self.logger.info(f"True labels and Predictions are {labels}, {pred} respectively. {v}")

        if self.args.return_entity_level_metrics:
            final_result = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for entity, score in value.items():
                        final_result[f"{key}_{entity}"] = score
                else:
                    final_result[key] = value
            return final_result

        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }


metric = evaluate.load("seqeval")


def compute_metrics_for_one_sentence(prediction, label, id_to_label, target_metric="f1"):
    prediction = np.argmax(prediction, axis=1)

    true_predictions = [id_to_label[p] for (p, l) in zip(prediction, label) if l != -100]
    true_labels = [id_to_label[l] for (p, l) in zip(prediction, label) if l != -100]

    if set(true_predictions + true_labels) == {"O"}:
        return {'overall_f1': 1.0}
    results = metric.compute(predictions=[true_predictions],
                             references=[true_labels],
                             zero_division=0)

    if target_metric in ["precision", "recall", "f1", "accuracy"]:
        return results[f"overall_{target_metric}"]
    elif target_metric in [i.split("-")[-1] for i in id_to_label.values()]:
        return results[target_metric]
    else:
        return results
