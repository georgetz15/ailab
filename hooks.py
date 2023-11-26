
class KeepActivations:
    """
    Forward hook to save the activations of
    one or more layers. Use with .register_forward_hook
    """
    def __init__(self):
        self.activations = []

    def reset(self):
        self.activations = []

    def forward_hook(self, model, x, y):
        self.activations.append(y.detach().cpu())
