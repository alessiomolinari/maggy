import inspect
import torch
import torch.nn as nn
import copy
import numpy as np
from torch.utils.data import DataLoader, Dataset


class Ablator:
    def __init__(self, model, dataset):
        # TODO maybe you can have a check that model must be a nn.Sequential, otherwise you make it sequential
        if type(model) != nn.Sequential:
            raise ValueError("Only nn.Sequential is supported as model type")
        self.model = model
        self.dataset = dataset

        self.state_dictionary = model.state_dict()

    def ablate_layers(self, idx_list, input_shape, infer_activation=False):
        if idx_list is None:
            return copy.deepcopy(self.model)
            # Why a copy? Because if you perform a multiple feature ablation without layer ablation you train on the
            #  same model over and over again.
        if type(idx_list) == int:
            idx_list = [idx_list]
        if type(idx_list) == set:
            idx_list = list(idx_list)
        elif type(idx_list) != list:
            raise TypeError(
                "idx_list should be an integer, a list or a set of integers",
                "but instead {0} (of type {1}) was passed".format(
                    idx_list, type(idx_list).__name__
                ),
            )

        new_modules = self.get_module_list(self.model)

        if infer_activation:
            activations_idx = []
            for idx in idx_list:
                if ((idx + 1) < len(new_modules)) and self._is_activation(
                    new_modules[idx + 1]
                ):
                    activations_idx.append(idx + 1)
            idx_list = idx_list + activations_idx
            idx_list = list(set(idx_list))

        ablated_modules = self.remove_modules(new_modules, idx_list)
        correct_modules = self.match_model_features(ablated_modules, input_shape)
        ablated_model = nn.Sequential(*correct_modules)

        return ablated_model

    @staticmethod
    def match_model_features(model_modules, input_shape):
        # TODO you have to do a lot of testing with different pytorch layers
        tensor_shape = (1,) + input_shape
        last_valid_out_features = tensor_shape[1]
        i = 0
        input_tensor = torch.rand(tensor_shape)
        anti_stuck_idx = 0

        while i < len(model_modules):
            layer = model_modules[i]

            try:
                output_tensor = layer(input_tensor)
                anti_stuck_idx = 0
                last_valid_out_features = output_tensor.shape[1]
                # print(layer, "\t\t", output_tensor.shape)
                i += 1
                input_tensor = output_tensor

            except RuntimeError:
                anti_stuck_idx += 1

                if anti_stuck_idx > 1:
                    raise RuntimeError(
                        "Ablation failed. Check again what modules you are ablating"
                    )

                layer_type = type(layer)
                layer_signature = inspect.signature(layer_type)
                parameters = dir(layer) & layer_signature.parameters.keys()
                new_args = dict()

                for key, value in layer.__dict__.items():
                    if key in parameters:
                        new_args[key] = value

                if "in_features" in new_args:
                    new_args["in_features"] = last_valid_out_features

                elif "in_channels" in new_args:
                    new_args["in_channels"] = last_valid_out_features

                # This new initialization is necessary because even if you change the shape of the layer,
                #  without initialization you don't have the correct number of weights
                model_modules[i] = layer_type(**new_args)
        return model_modules

    def execute_trials(self):
        for i, trial in enumerate(self.trials):
            print("Starting trial", i)

            original_data = self.dataset.data

            # 1) Ablate layers
            ablated_model = self.ablate_layers(
                trial.ablated_layers, trial.input_shape, trial.infer_activation
            )

            # 2) Ablate features
            if trial.ablated_features is not None:
                print("Ablating features:", trial.ablated_features)
                self.dataset.ablate_feature(trial.ablated_features)

            # 3) Match features in model
            self.match_model_features(ablated_model, trial.input_shape)

            # 4) Train
            dataloader = DataLoader(self.dataset, **self.dataloader_kwargs)
            trial.metric = self.training_fn(ablated_model, dataloader)
            print("Final metric:", trial.metric, "\n\n")

            # 5) Restore original data
            self.dataset.data = original_data

    @staticmethod
    def get_module_list(model):
        modules = []
        for mod in model.modules():
            # TODO this is just a quick patch but you should find a better solution
            if not str(mod).startswith("Sequential"):
                modules.append(mod)
        # In PyTorch the first module is actually a description of the whole model
        # TODO the following two lines interfere with the condition on Sequential
        #  gotta find a better way to generalize complex models
        # removed = modules.pop(0)
        # print("This is removed\n", removed)
        return modules

    def remove_modules(self, modules_list, modules_to_ablate):
        for i in reversed(sorted(modules_to_ablate)):
            self._ablate_and_print(modules_list, i)
        return modules_list

    @staticmethod
    def _ablate_and_print(modules, i):
        ablated = modules.pop(i)
        print("Ablating layer ", i, " - ", ablated, sep="")

    # TODO static methods might be probably refactored to a module utils.py
    @staticmethod
    def _is_activation(layer):
        from torch.nn.modules import activation

        activation_functions = inspect.getmembers(activation, inspect.isclass)
        activation_functions = [x[0] for x in activation_functions]
        if layer.__class__.__name__ in activation_functions:
            return True
        else:
            return False


class MaggyDataset(Dataset):
    """
    In PyTorch there is no way to get the entire dataset starting from the classes Dataset or DataLoader.
     This is because the only method whose implementation is guaranteed is __getitem__ (enumerate) but there is
     no specification on what this method should return. For instance, it could return a row of a tabular dataset,
     as well as a tuple (label, row). For this reason we necessitate a method that returns a tabular dataset
    (tabular because we define feature ablation only on tabular datasets for now) on which we can ablate the columns.
    """

    def __init__(self, data):
        self.data = data

    def ablate_feature(self, feature):
        raise NotImplementedError


class PandasDataset(Dataset):
    def __init__(self, df, label):
        super(Dataset).__init__()
        from pyspark.sql.dataframe import DataFrame

        if isinstance(df, DataFrame):
            df = df.toPandas()
        self.label = label
        self.columns = df.columns.values.tolist()

        if self.label not in self.columns:
            raise ValueError(
                "The specified label can't be found among the dataset columns"
            )

        label_idx = self.columns.index(label)
        self.columns.pop(label_idx)
        self.data = np.delete(df.values, obj=label_idx, axis=1)
        self.labels = df.values[:, label_idx]

    def __getitem__(self, item):
        return self.labels[item], self.data[item]

    def __len__(self):
        return self.data.shape[0]

    def ablate_feature(self, feature):
        lowercase_columns = [col.lower() for col in self.columns]
        feature = feature.lower()
        if feature.lower() in lowercase_columns:
            idx = lowercase_columns.index(feature)
            self.data = np.delete(self.data, obj=idx, axis=1)
        else:
            raise RuntimeError("Ablation failed: column not found")


class Trial:
    def __init__(self, input_shape, ablated_layers, ablated_features, infer_activation):
        self.ablated_layers = ablated_layers
        self.ablated_features = ablated_features
        self.input_shape = input_shape
        self.infer_activation = infer_activation
        self.metric = None
