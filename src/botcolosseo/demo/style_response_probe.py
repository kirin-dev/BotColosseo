"""Read-only condition interventions at actual high-level decision states."""

import torch

from botcolosseo.demo.hierarchical_controls import ScheduledController


class ProbedScheduledController(ScheduledController):
    def reset(self):
        super().reset()
        self.style_probes = []

    @torch.no_grad()
    def step(self, *args, **kwargs):
        record = super().step(*args, **kwargs)
        if self.last_high_inputs is not None:
            inputs = dict(self.last_high_inputs)
            probabilities = {}
            for name, style in (("neutral", (0, 0, 0)), ("aggressive", (1, 0, 0)),
                                ("defensive", (0, 1, 0)), ("explorer", (0, 0, 1))):
                inputs["style"] = torch.tensor([[style]], dtype=torch.float32, device=self.device)
                probabilities[name] = self.strategy(**inputs).logits[0, 0].softmax(-1).tolist()
            self.style_probes.append({"decision": record["decision"],
                                      "command_probabilities": probabilities})
        return record
