import json

config_file_path = "/home/sowmya_kanuri/PycharmProjects/receipt_transaction_matching/config_folder/config_file.json"

class Config():
    '''
    config
    '''

    __instance = None

    @staticmethod
    def get_instance():
        """ Static access method. """
        if Config.__instance is None:
            Config()
        return Config.__instance

    def __init__(self):
        if Config.__instance is not None:
            raise Exception("ALready initialized config file")
        else:
            self.config = self.read_config_file()
            Config.__instance = self

    def read_config_file(self):
        '''
        method
        '''
        with open(config_file_path, mode='r') as json_data_file:
            self.config = json.load(json_data_file)
        return self.config
