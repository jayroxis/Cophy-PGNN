def getPresets(DNN_type):
    train_loss_list = []
    test_loss_list = []
    norm = True
    
    if DNN_type == "PGNN_":
        train_loss_list = ['mse_loss',
            'phy_loss', 
            'energy_loss']
        test_loss_list = [
            'phy_loss', 
            'energy_loss']
        norm = True
    elif DNN_type == "PGNN_LF":
        train_loss_list = ['phy_loss', 'energy_loss']
        test_loss_list = ['phy_loss', 'energy_loss']
        norm = True  
    elif DNN_type == "PGNN_OnlyDTr":
        train_loss_list = ['mse_loss', 'phy_loss', 'energy_loss']
        test_loss_list = []
        norm = True
    elif DNN_type=="NN":
        train_loss_list = ['mse_loss']
        test_loss_list = []
        norm = False
    else:
        print("Unknown ANN type!", DNN_type)
    return (train_loss_list, test_loss_list, norm)