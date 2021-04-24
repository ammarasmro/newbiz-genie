from name_generator.models.model import Model


class LyModel(Model):
    
    def fit(self, data):
        pass

    def predict(self, data):
        return f'{data}ly'
