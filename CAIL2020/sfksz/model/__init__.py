from model.qa.qa import Model,CAPSModel

model_list = {
    "Model": Model,
    "CAPSModel": CAPSModel
}


def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError
