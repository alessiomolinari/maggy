from maggy.ablation.ablator import AbstractAblator
from hops import featurestore
from maggy.trial import Trial
from .pytorch_ablator import Ablator, PandasDataset


class LOCOPyTorch(AbstractAblator):
    def __init__(self, ablation_study, final_store):
        super().__init__(ablation_study, final_store)
        self.base_model_generator = self.ablation_study.model.base_model_generator
        self.base_dataset_generator = self.get_dataset_generator(ablated_feature=None)

    def get_number_of_trials(self):
        return (
            len(self.ablation_study.features.included_features)
            + len(self.ablation_study.model.layers.included_layers)
            + len(self.ablation_study.model.layers.included_groups)
            + len(self.ablation_study.model.custom_model_generators)
            + 1
        )

    def get_dataset_generator(self, ablated_feature=None, dataset_type="numpy"):
        training_dataset_name = self.ablation_study.hops_training_dataset_name
        training_dataset_version = self.ablation_study.hops_training_dataset_version
        label_name = self.ablation_study.label_name

        def create_dataset():
            pandas_df = featurestore.get_training_dataset(
                training_dataset_name,
                training_dataset_version=training_dataset_version,
                dataframe_type="pandas",
            )

            dataset = PandasDataset(pandas_df, label_name)

            if ablated_feature is not None:
                dataset.ablate_feature(ablated_feature)

            return dataset

        return create_dataset

    def get_model_generator(self, layer_identifier=None):
        if layer_identifier is None:
            return self.base_model_generator

        dataset_fn = self.get_dataset_generator(ablated_feature=None)

        ablator = Ablator(self.base_model_generator(), dataset=dataset_fn())

        def model_generator():
            # TODO input_shape must be changed if the dataset is not tabular
            num_columns = ablator.dataset.data.shape[1]
            input_shape = (num_columns,)

            ablated_model = ablator.ablate_layers(
                layer_identifier, input_shape, infer_activation=False
            )

            # model = Ablator.match_model_features(ablated_model, input_shape)
            return ablated_model

        return model_generator

    def initialize(self):
        """
        Prepares all the trials for LOCO policy (Leave One Component Out).
        In total `n+1` trials will be generated where `n` is equal to the number of components
        (e.g. features and layers) that are included in the ablation study
        (i.e. the components that will be removed one-at-a-time). The first trial will include all the components and
        can be regarded as the base for comparison.
        """

        # 0 - add first trial with all the components (base/reference trial)
        self.trial_buffer.append(
            Trial(self.create_trial_dict(None, None), trial_type="ablation")
        )

        # generate remaining trials based on the ablation study configuration:
        # 1 - generate feature ablation trials
        for feature in self.ablation_study.features.included_features:
            self.trial_buffer.append(
                Trial(
                    self.create_trial_dict(ablated_feature=feature),
                    trial_type="ablation",
                )
            )

        # 2 - generate single-layer ablation trials
        for layer in self.ablation_study.model.layers.included_layers:
            self.trial_buffer.append(
                Trial(
                    self.create_trial_dict(layer_identifier=layer),
                    trial_type="ablation",
                )
            )

        # 3 - generate layer-groups ablation trials
        # each element of `included_groups` is a frozenset of a set, so we cast again to get a set
        # why frozensets in the first place? because a set can only contain immutable elements
        # hence elements (layer group identifiers) are frozensets

        for layer_group in self.ablation_study.model.layers.included_groups:
            self.trial_buffer.append(
                Trial(
                    self.create_trial_dict(layer_identifier=set(layer_group)),
                    trial_type="ablation",
                )
            )

        # 4 - generate ablation trials based on custom model generators

        for custom_model_generator in self.ablation_study.model.custom_model_generators:
            self.trial_buffer.append(
                Trial(
                    self.create_trial_dict(
                        custom_model_generator=custom_model_generator
                    ),
                    trial_type="ablation",
                )
            )

    def get_trial(self, ablation_trial=None):
        if self.trial_buffer:
            return self.trial_buffer.pop()
        else:
            return None

    def finalize_experiment(self, trials):
        pass

    def create_trial_dict(
        self, ablated_feature=None, layer_identifier=None, custom_model_generator=None
    ):
        """
        Creates a trial dictionary that can be used for creating a Trial instance.

        :param ablated_feature: a string representing the name of a feature, or None
        :param layer_identifier: A string representing the name of a single layer, or a set representing a layer group.
        If the set has only one element, it is regarded as a prefix, so all layers with that prefix in their names
        would be regarded as a layer group. Otherwise, if the set has more than one element, the layers with
        names corresponding to those elements would be regarded as a layer group.
        :return: A trial dictionary that can be passed to maggy.Trial() constructor.
        :rtype: dict
        """

        trial_dict = {}

        # 1 - determine the dataset generation logic
        if ablated_feature is None:
            trial_dict["dataset_function"] = self.base_dataset_generator
            trial_dict["ablated_feature"] = "None"
        else:
            trial_dict["dataset_function"] = self.get_dataset_generator(
                ablated_feature, dataset_type="tfrecord"
            )
            trial_dict["ablated_feature"] = ablated_feature

        # 2 - determine the model generation logic
        # 2.1 - no model ablation

        if layer_identifier is None and custom_model_generator is None:
            trial_dict[
                "model_function"
            ] = self.ablation_study.model.base_model_generator
            trial_dict["ablated_layer"] = "None"
        # 2.2 - layer ablation based on base model generator
        elif layer_identifier is not None and custom_model_generator is None:
            trial_dict["model_function"] = self.get_model_generator(
                layer_identifier=layer_identifier
            )
            if type(layer_identifier) is str or type(layer_identifier) is int:
                trial_dict["ablated_layer"] = layer_identifier
            elif type(layer_identifier) is set:
                if len(layer_identifier) > 1:
                    trial_dict["ablated_layer"] = str(list(layer_identifier))
                elif len(layer_identifier) == 1:
                    trial_dict["ablated_layer"] = "Layers prefixed " + str(
                        list(layer_identifier)[0]
                    )
        # 2.3 - model ablation based on a custom model generator
        elif layer_identifier is None and custom_model_generator is not None:
            trial_dict["model_function"] = self.get_model_generator(
                custom_model_generator
            )
            trial_dict["ablated_layer"] = "Custom model: " + custom_model_generator[1]

        return trial_dict

    # TODO method create_trial.dict is not in the abstract ablator, is it necessary?
