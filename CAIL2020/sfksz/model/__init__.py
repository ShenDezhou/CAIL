from model.qa.qa import Model,GRUModel

model_list = {
    "Model": Model,
    "GRUModel": GRUModel
}


def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError
