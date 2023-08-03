import hashlib
import pickle
from pathlib import Path

class ModelInfo:
    def __init__(self, filename=None) -> None:
        # self.hf_models = [
        #     'huggyllama/llama-7b',
        #     'huggyllama/llama-13b',
        #     'huggyllama/llama-30b',
        #     'meta-llama/Llama-2-7b-hf',
        #     'meta-llama/Llama-2-13b-hf',
        #     'meta-llama/Llama-2-7b-chat-hf',
        #     'meta-llama/Llama-2-13b-chat-hf',
        #     'lmsys/vicuna-7b-v1.3',
        #     'distilgpt2',
        # ]
        self.filename = filename
        if self.filename is None or not Path(self.filename).is_file():
            self.hf_models = []
        else:
             with open(self.filename, 'rb') as file:
                self.hf_models = pickle.load(file)

    @staticmethod
    def get_hashed_modelname(model_name: str):
        return hashlib.md5(model_name.encode('utf-8')).hexdigest()
    
    def get_modelname(self, hash_md5):
        return self.list_hash_to_modelname()[hash_md5]
    
    def add_model(self, model_name:str, exist_ok=True):
        if not exist_ok and model_name in set(self.hf_models):
            raise Exception('model already added')
        if not model_name in set(self.hf_models):
            self.hf_models.append(model_name)
        if self.filename is not None:
            with open(self.filename, 'wb') as file:
                pickle.dump(self.hf_models, file)

    def list_modelname_to_hash(self):
        return \
        {
            modelname: self.get_hashed_modelname(modelname) for modelname in self.hf_models
        }
    
    def list_hash_to_modelname(self):
        return dict((y, x) for x, y in self.list_modelname_to_hash().items())

MODELNAMES_PKL= Path(__file__).parent.parent / '.modelnames.pkl'
model_info = ModelInfo(MODELNAMES_PKL)
if __name__ == "__main__":
    # model_info = ModelInfo()
    # print(model_info.list_modelname_to_hash())
    print(model_info.get_hashed_modelname('msys/vicuna-13b-v1.3'))
    # print(model_info.get_modelname('fc433f70103338181ac914a44eb2749c'))

    # # adds existed model
    # try:
    #     model_info.add_model('huggyllama/llama-7b', exist_ok=False)
    # except Exception as e:
    #     print(e)


    preset_hf_models = [
        'huggyllama/llama-7b',
        'huggyllama/llama-13b',
        'huggyllama/llama-30b',
        'meta-llama/Llama-2-7b-hf',
        'meta-llama/Llama-2-13b-hf',
        'meta-llama/Llama-2-7b-chat-hf',
        'meta-llama/Llama-2-13b-chat-hf',
        'lmsys/vicuna-7b-v1.3',
        'lmsys/vicuna-13b-v1.3',
        'distilgpt2',
    ]

    for m in preset_hf_models:
        model_info.add_model(m)
        print(model_info.list_modelname_to_hash())

