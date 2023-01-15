import gc
import logging

import model
from utils import *

gc.collect()

if __name__ == "__main__":
    mode = input("Choose between train or test: ")
    if mode.lower() == "test":
        model_path = input("Input the path of the model you want to load: ")
        net = model.Net()

        if torch.cuda.is_available():
            net.cuda()

        net.load(path=r'{}'.format(model_path), slim=True)

        results_dest = default_test_results_dir()
        if not os.path.isdir(results_dest):
            os.makedirs(results_dest)

        image_path = input("Input the path of the image you want to modify: ")
        age = int(input("Input the target age: "))
        gender = int(input("Input the target gender(0 - male, 1 - female): "))
        net.test(image_path=r'{}'.format(image_path), age=age, gender=gender, target=results_dest)
    else:
        data_src = consts.UTKFACE_DEFAULT_PATH
        print("Data folder is {}".format(data_src))
        results_dest = default_train_results_dir()
        os.makedirs(results_dest, exist_ok=True)
        print("Results folder is {}".format(results_dest))

        log_path = os.path.join(results_dest, 'log_results.log')
        if os.path.exists(log_path):
            os.remove(log_path)
        logging.basicConfig(filename=log_path, level=logging.DEBUG)

        net = model.Net()

        if torch.cuda.is_available():
            net.cuda()

        epochs = int(input("Input the epoch count: "))

        net.start_training(
            dataset_path=data_src,
            epochs=epochs,
            where_to_save=results_dest
        )
