from utils.config import parse_args
from utils.data_loader import get_data_loader

from models.gan import GAN
from models.cgan import CGAN
from models.dcgan import DCGAN_MODEL

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def main(args):
    model = None
    if args.model == 'GAN':
        model = GAN(args)
    elif args.model == 'DCGAN':
        model = DCGAN_MODEL(args)
    elif args.model == 'CGAN':
        model = CGAN(args)
    else:
        print("Model type non-existing. Try again.")
        exit(-1)

    # Load datasets to train and test loaders
    train_loader, test_loader = get_data_loader(args)



    # Start model training
    if args.is_train == 'True':
        model.train(train_loader)




    # start evaluating on test data
    else:
        model.evaluate(test_loader, args.load_D, args.load_G)
        # for i in range(50):
        #    model.generate_latent_walk(i)


if __name__ == '__main__':
    args = parse_args()
    print(args.cuda)
    main(args)
