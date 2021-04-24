def load_args():
    parser = argparse.ArgumentParser(description="Parsing arguments")

    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')

    args = parser.parse_args()
    return args

class Trainer(object):
    def __init__(self, args):
        self.args = args

    def training(self, epoch):
        train_loss = 0.0

    def validation(self, epoch):
        self.model.eval()

if __name__ == "__main__":

    args = load_args()

    if args.debug:
        print("Wait for debugger attach")
        import debugpy
        # https://code.visualstudio.com/docs/python/debugging#_remote-debugging
        # Launch applicaiton on remote computer: 
        # > python3 -m ptvsd --host 10.150.41.30 --port 3000 --wait fcn/train.py
        # Allow other computers to attach to ptvsd at this IP address and port.
        debugpy.listen(address=('0.0.0.0', args.debug_port))
        # Pause the program until a remote debugger is attached

        debugpy.wait_for_client()
        print("Debugger attached")
    
    trainer = Trainer(args)
    print('Starting epoch {} of {}'.format(trainer.args.start_epoch,  trainer.args.epochs)
    print('Total Epoches:',)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val:
            trainer.validation(epoch)

    trainer.writer.close()