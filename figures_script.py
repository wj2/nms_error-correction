
import mixedselectivity_theory.figures as nmf

if __name__ == '__main__':
    nmf.figure1()
    nmf.figure2()
    nmf.figure3()

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
    nmf.figure4(data_paths=dps)
