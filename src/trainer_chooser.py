import model.trainer

import pdb
def trainer_chooser(args):
  trainer_name = 'model.trainer.Trainer'
  if 'Joint' in args.model:
    trainer_name += 'Joint'
  if 'Late' in args.model:
    trainer_name += 'Late'
  if 'Gest' in args.model:
    trainer_name += 'Gest'
  if 'Cluster' in args.model:
    trainer_name += 'Cluster'
  if args.gan:
    trainer_name += 'GAN'
  if 'Classifier' in args.model:
    trainer_name += 'Classifier'

  try:
    eval(trainer_name)
  except:
    raise '{} trainer not defined'.format(trainer_name)
  print('{} selected'.format(trainer_name))
  return eval(trainer_name)
