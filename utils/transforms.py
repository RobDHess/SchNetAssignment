import esm
import torch


models = {"esm2_t6_8M_UR50D": (esm.pretrained.esm2_t6_8M_UR50D, 6)}


class ESMTransform:
    def __init__(self, name, device):
        assert name in models.keys(), "Model not recognized"

        model_fn, layers = models[name]
        self.model, self.alphabet = model_fn()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval()  # disables dropout for deterministic results
        self.device = device
        self.layers = layers
        self.name = name

        self.model = self.model.to(self.device)

    def __call__(self, data):
        sequence = data.sequence
        protein_data = [("protein", sequence)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(protein_data)

        batch_tokens = batch_tokens.to(self.device)

        with torch.no_grad():
            results = self.model(
                batch_tokens, repr_layers=[self.layers], return_contacts=True
            )
        embeddings = results["representations"][self.layers]

        data.x = embeddings[0, 1:-1, :].cpu()
        return data
