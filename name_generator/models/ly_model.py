from name_generator.models.model import Model


class LyModel(Model):

    def fit(self, data):
        pass

    def predict(self, data):
        if data:
            first_word = data.split(' ')[:1]
        else:
            data = '-'
        return f'{data}ly'
