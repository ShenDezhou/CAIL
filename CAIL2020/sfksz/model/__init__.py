from model.qa.qa import Model, ModelXS,ModelS, ModelX, BertCAPSModel

model_list = {
    "Model": Model,
    "ModelXS": ModelXS,
    "ModelS": ModelS,
    "ModelX": ModelX,
    "BertCAPSModel": BertCAPSModel
}


def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError
