# extract the DNA of a model and back
from keras.models import clone_model


class KerasWrapper():
    def __init__(self, model, mutation_func):
        self.model = model
        self.mutation = mutation_func

    def mutateWith(self, lover):
        for i in range(len(self.model.layers)):
            layersA = self.model.layers[i].get_weights()
            layersB = lover.model.layers[i].get_weights()
            for j in range(len(layersA)):
                layersA[j] = self.mutation(layersA[j], layersB[j])
                print layersA[j]

            self.model.layers[i].set_weights(layersA)

    def new(self):
        mo = clone_model(self.model)
        return KerasWrapper(mo, self.mutation)
