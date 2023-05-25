import torch

from synthesizers.pattern_synthesizer import PatternSynthesizer


class ComplexSynthesizer(PatternSynthesizer):
    """
    For physical backdoors it's ok to train using pixel pattern that
    represents the physical object in the real scene.
    """

    pattern_tensor: torch.Tensor = torch.tensor([
        [-10, 1, 100, 1, -10],
        [1, 10., 200, 10, 1],
        [100, 200., 200, 200, 100],
        [1, 10., 200, 10, 1],
        [-10, 10, 100, 10, -10]
    ])

    def synthesize_labels(self, batch, attack_portion=None):
        # TODO: implement multiplication/addition labels
        batch.labels[:attack_portion] = batch.aux[:attack_portion]
        return
