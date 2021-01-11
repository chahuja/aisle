import model.trainer

import pdb
def trainer_chooser(args):
  trainer_name = 'model.trainer.Trainer'
  if args.noise_only:
    trainer_name += 'NoiseOnly'
  if 'Joint' in args.model:
    trainer_name += 'Joint'
  if 'Late' in args.model:
    trainer_name += 'Late'
  if 'Gest' in args.model:
    trainer_name += 'Gest'
  # if 'Transformer' in args.model:
  #   trainer_name += 'Transformer'
  if 'Cluster' in args.model:
    trainer_name += 'Cluster'
  if 'Style' in args.model:
    trainer_name += 'Style'
  if 'Disentangle' in args.model:
    trainer_name += 'Disentangle'
  if 'Learn' in args.model:
    trainer_name += 'Learn'
  if args.pos:
    trainer_name += 'POS'
  if 'Contrastive' in args.model:
    trainer_name += 'Contrastive'
  if args.gan:
    trainer_name += 'GAN'
  # if args.sample_all_styles:
  #   trainer_name += 'Sample'
  if args.mix:
    trainer_name += 'Mix'
  if 'NN' in args.model:
    trainer_name += 'NN'
  if 'Rand' in args.model:
    trainer_name += 'Rand'
  if 'Mean' in args.model:
    trainer_name += 'Mean'
  if 'Classifier' in args.model:
    trainer_name += 'Classifier'

  try:
    eval(trainer_name)
  except:
    raise '{} trainer not defined'.format(trainer_name)
  print('{} selected'.format(trainer_name))
  return eval(trainer_name)
