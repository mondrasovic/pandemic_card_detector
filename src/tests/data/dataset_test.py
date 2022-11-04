import collections
import itertools

import pytest

from core.data.dataset import SubsetType, SubsetTypeRandomGen


@pytest.mark.usefixtures("fix_rand_seed")
class TestSubsetTypeRandomGen:
    @pytest.mark.parametrize(
        "val_portion, test_portion",
        [
            pytest.param(-1, 0, id="negative_portion"),
            pytest.param(0.8, 0.8, id="portion_sum_exceeds_one"),
        ],
    )
    def test_raises_if_subset_portion_invalid(self, val_portion, test_portion):
        with pytest.raises(ValueError):
            SubsetTypeRandomGen(
                dataset_size=100, val_portion=val_portion, test_portion=test_portion
            )

    @pytest.mark.parametrize(
        "dataset_size, val_portion, test_portion, expected_train_val_test_samples_count",
        [
            pytest.param(1000, 0.2, 0.2, (600, 200, 200), id="60/20/20_split"),
            pytest.param(100, 0.1, 0.1, (80, 10, 10), id="80/10/10_split"),
            pytest.param(50, 0.0, 0.0, (50, 0, 0), id="train_only"),
            pytest.param(50, 1.0, 0.0, (0, 50, 0), id="val_only"),
            pytest.param(50, 0.0, 1.0, (0, 0, 50), id="test_only"),
        ],
    )
    def test_subset_counts_reflect_portions(
        self,
        dataset_size,
        val_portion,
        test_portion,
        expected_train_val_test_samples_count,
    ):
        subset_type_random_gen = SubsetTypeRandomGen(dataset_size, val_portion, test_portion)
        generated_subset_types = list(iter(subset_type_random_gen))
        assert len(generated_subset_types) == dataset_size

        subset_type_counter = collections.Counter(generated_subset_types)

        assert subset_type_counter[SubsetType.TRAIN] == expected_train_val_test_samples_count[0]
        assert subset_type_counter[SubsetType.VAL] == expected_train_val_test_samples_count[1]
        assert subset_type_counter[SubsetType.TEST] == expected_train_val_test_samples_count[2]

    @pytest.mark.parametrize(
        "dataset_size, val_portion, test_portion",
        [
            pytest.param(400, 0.2, 0.2, id="60/20/20_split"),
            pytest.param(100, 0.1, 0.1, id="80/10/10_split"),
            pytest.param(600, 0.3, 0.3, id="40/30/30_split"),
        ],
    )
    def test_generates_shuffled_sequence(self, dataset_size, val_portion, test_portion):
        subset_type_random_gen = SubsetTypeRandomGen(dataset_size, val_portion, test_portion)
        generated_subset_types = list(iter(subset_type_random_gen))

        # Some trivial checks that the sequence differs from all possible results of sorting
        for subset_type_order_permutation in itertools.permutations(
            (SubsetType.TRAIN, SubsetType.VAL, SubsetType.TEST)
        ):
            subset_types_sorted = sorted(
                generated_subset_types,
                key=lambda item: subset_type_order_permutation.index(item),
            )
            assert generated_subset_types != subset_types_sorted
