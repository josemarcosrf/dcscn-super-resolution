import os
import torch


def save_model(model, save_path, checkpoint_name):
    """
    Saves the state_dict of a PyTorch model
    Args:
        model (PyTorch.nn.Module): model to save its state_dict
        save_path (str): directory where to save the state_dict
        checkpoint_name (str): state_dict checkpoint file name
    """
    chkpt_path = os.path.join(save_path, checkpoint_name)
    torch.save(model.state_dict(), chkpt_path)
    # save reference to best checkpoint
    with open(os.path.join(save_path, 'checkpoint.txt'), "w") as fh:
        fh.write(checkpoint_name)


def load_model(model, save_path, checkpoint_name):
    """
    Loads into an existing build PyTorch model the state_dict
    Args:
        model (PyTorch.nn.Module): model to load the state_dict into
        save_path (str): directory where to find the state_dict
        checkpoint_name (str): state_dict checkpoint file name

    Returns:
        PyTorch.nn.Module model: model with the state dict loaded
        from the specified checkpoint
    """
    # load the just trained model
    chkpt_path = os.path.join(save_path, checkpoint_name)
    model.load_state_dict(
        torch.load(
            chkpt_path,
            map_location=lambda storage, loc: storage
        )
    )
    return model
