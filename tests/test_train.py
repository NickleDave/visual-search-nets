import os
import unittest
import tempfile
import shutil
from configparser import ConfigParser

import searchnets
import searchnets.__main__

HERE = os.path.dirname(__file__)
TEST_DATA_DIR = os.path.join(HERE, 'test_data')
TEST_CONFIGS_DIR = os.path.join(TEST_DATA_DIR, 'configs')


class TestTrain(unittest.TestCase):
    def setUp(self):
        self.tmp_output_dir = tempfile.mkdtemp()
        # below, note that copyfile returns destination path
        src = os.path.join(TEST_CONFIGS_DIR, 'searchnets_feature_search_alexnet.ini')
        dst = os.path.join(self.tmp_output_dir, 'searchnets_feature_search_alexnet.ini')
        self.tmp_config_file = shutil.copyfile(src, dst)
        self.config_obj = ConfigParser()
        self.config_obj.read(self.tmp_config_file)
        self.config_obj['DATA']['GZ_FILENAME'] = os.path.join(self.tmp_output_dir,
                                                              'data_prepd_for_nets',
                                                              'feature_search_alexnet_data.gz')
        self.config_obj['TRAIN']['MODEL_SAVE_PATH'] = os.path.join(self.tmp_output_dir,
                                                                   'checkpoints',
                                                                   'feature_search_alexnet_models')
        with open(self.tmp_config_file, 'w') as fp:
            self.config_obj.write(fp)

        self.config_obj = searchnets.config.parse_config(self.tmp_config_file)
        searchnets.__main__._call_data(self.config_obj)

    def tearDown(self):
        shutil.rmtree(self.tmp_output_dir)

    def test_train_runs_without_crash(self):
        config = self.config_obj
        searchnets.train(gz_filename=config.data.gz_filename,
                         net_name=config.train.net_name,
                         number_nets_to_train=config.train.number_nets_to_train,
                         input_shape=config.train.input_shape,
                         new_learn_rate_layers=config.train.new_learn_rate_layers,
                         base_learning_rate=config.train.base_learning_rate,
                         new_layer_learning_rate=config.train.new_layer_learning_rate,
                         epochs_list=config.train.epochs_list,
                         batch_size=config.train.batch_size,
                         random_seed=config.train.random_seed,
                         model_save_path=config.train.model_save_path,
                         dropout_rate=config.train.dropout_rate,
                         val_size=config.data.val_size)


if __name__ == '__main__':
    unittest.main()
