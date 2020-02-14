import utils as ut

inputs, _ = ut.dataset.get_celeba_data(ut.general_constants.datapath, group_num=2)

print(inputs.shape)

ut.BetaTCVAE()