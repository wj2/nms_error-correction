
import mixedselectivity_theory.figures as nmf

if __name__ == '__main__':
    basefolder = './' # this is where the figure files will be saved
    
    nmf.figure1(basefolder=basefolder)
    nmf.figure2(basefolder=basefolder)
    nmf.figure3(basefolder=basefolder)

    # datapath_maceo = 'lip_saccdmc/dmc/lip/maceo/'
    # datapath_jb = 'lip_saccdmc/dmc/lip/jb/'
    # dps = (datapath_maceo, datapath_jb)
    dps = []
    if datapath_maceo is not None:
        dps.append(datapath_maceo)
    if datapath_jb is not None:
        dps.append(datapath_jb)
    if len(dps) == 0:
        dps = None
    nmf.figure4(basefolder=basefolder, data_paths=dps)
