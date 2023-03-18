"""
a class to load trajectory and interpolate the data for DMP
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Todo: 1. load different number trajectory
#       2. create standard pytorch dataloader
#       3. data augmentation using classical DMPS (generate different goals)


class DataLoader(object):
    def __init__(self, run_time, dt, dof):
        self.run_time = run_time
        self.dt = dt
        self.time_steps = np.arange(0, self.run_time, self.dt)
        self.dof = dof
        self.paths = []

    def load_data(self, data_file):
        """
        load a data in [steps, dof] format
        """
        traj = np.load(data_file)["arr_0"]
        traj_time = np.linspace(0, self.run_time, traj.shape[0])
        assert traj.shape[1] == self.dof, "dof of data is not equal to the dof of DMP system"

        new_traj = np.zeros((len(self.time_steps), self.dof))
        for _d in range(self.dof):
            path_gen = interp1d(traj_time, traj[:, _d])
            new_traj[:, _d] = path_gen(self.time_steps)
        self.paths.append(new_traj)


# test the data loader
if __name__ == "__main__":
    # %% load data
    data_file = "2.npz"
    traj = np.load(data_file)["arr_0"]
    plt.figure(1, figsize=(6, 6))
    plt.plot(traj[:, 0], traj[:, 1], "b")
    plt.title("DMP system - draw number 2")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.show()

    # test data loader class
    traj_loader = DataLoader(run_time=1.0, dt=0.01, dof=2)
    traj_loader.load_data(data_file)
    y_inter = traj_loader.paths[0]
    plt.figure(2, figsize=(6, 6))
    plt.plot(y_inter[:, 0], y_inter[:, 1], "b")
    plt.title("DMP system - draw number 2 -interpolated")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.show()
