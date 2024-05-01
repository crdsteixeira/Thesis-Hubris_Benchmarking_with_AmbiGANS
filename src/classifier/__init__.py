from .simple_cnn import Classifier as SimpleCNN
from .my_mlp import Classifier as MyMLP
from .ensemble import Ensemble as Ensemble
from .classifier_cache import ClassifierCache


def construct_classifier(params, device=None):
    if params['type'] == 'cnn':
        C = SimpleCNN(params['img_size'], [params['nf'], params['nf'] * 2],
                      params['n_classes'])
    elif params['type'] == 'mlp':
        C = MyMLP(params['img_size'], params['n_classes'], [params['nf'], params['nf'] * 2])
    elif params['type'].split(':')[0] == 'ensemble':
        ensemble_type = params['type'].split(':')[1]
        output_method = params['type'].split(':')[2]
        C = Ensemble(params['img_size'], params['n_classes'], params['nf'], ensemble_type, output_method, device=device)
    else:
        exit(-1)

    return C.to(device)
