class ModelInfo:
    def __init__(self) -> None:
        pass

    @staticmethod
    def encode_modelname(model_name):
        """encode to avoid / character in model name which is invalid character of directory name"""
        encoded = model_name.replace(".", "_")
        encoded = encoded.replace("/", ".")
        return encoded

    @staticmethod
    def get_decoded_modelname(encoded_model_name):
        decoded = encoded_model_name.replace(".", "/")
        decoded = decoded.replace("_", ".")
        return decoded

if __name__ == "__main__":
    print(ModelInfo.encode_modelname('msys/vicuna-13b-v1.3'))
    print(ModelInfo.get_decoded_modelname('msys.vicuna-13b-v1_3'))
    model_id = 'msys/vicuna-13b-v1.3'
    assert model_id==ModelInfo.get_decoded_modelname(ModelInfo.encode_modelname(model_id))