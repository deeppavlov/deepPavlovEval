from deeppavlov.models.embedders.elmo_embedder import ELMoEmbedder as ELMoEmbedder_hub

class ELMoEmbedder:
    def __init__(self, spec_url=None, elmo_output_names=None, mean=False):
        if spec_url is None:
            spec_url = "http://files.deeppavlov.ai/deeppavlov_data/elmo_ru-wiki_600k_steps.tar.gz"
        if mean:
            if elmo_output_names is not None:
                raise ValueError('elmo output names should be None if mean==True')
            elmo_output_names = ['default']
        if elmo_output_names is None:
            elmo_output_names = ['elmo']

        self.mean = mean
        self.model = ELMoEmbedder_hub(spec_url, elmo_output_names=elmo_output_names)
        self.dim = self.model.dim

    def __call__(self, batch, mean=None):
        return self.model(batch)
