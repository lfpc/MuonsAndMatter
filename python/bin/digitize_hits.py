import h5py
import numpy as np
import os
import torch
import time


def read_data(file_path):
    """
    Reads data from an HDF5 file and returns the relevant tensors.
    """
    with h5py.File(file_path, "r") as f:
        px = torch.tensor(np.concatenate(f["px"][:]), dtype=torch.float32)
        py = torch.tensor(np.concatenate(f["py"][:]), dtype=torch.float32)
        pz = torch.tensor(np.concatenate(f["pz"][:]), dtype=torch.float32)
        x = torch.tensor(np.concatenate(f["x"][:]), dtype=torch.float32)
        y = torch.tensor(np.concatenate(f["y"][:]), dtype=torch.float32)
        z = torch.tensor(np.concatenate(f["z"][:]), dtype=torch.float32)
        pdg_id = torch.tensor(np.concatenate(f["pdg_id"][:]), dtype=torch.int64)

    return px, py, pz, x, y, z, pdg_id

class Tagger:
    dY1 = 1.35
    dY2 = 3.0
    dX1 = 0.5
    dX2 = 2.0
    length = torch.tensor(50.0)
    z0 = 33.12-2.14

    dX_ubt = 1.2 / 2
    x0_ubt = -dX_ubt
    x1_ubt = dX_ubt

    dY_ubt = 1.5/2
    y0_ubt = -dY_ubt
    y1_ubt = dY_ubt


    def __init__(self, grid_ubt=(50, 1), grid_sbt=(5, 8, 62)):
        self.grid_ubt = grid_ubt
        self.grid_sbt = grid_sbt

        self.dx_UBT = (self.x1_ubt - self.x0_ubt) / grid_ubt[0]
        self.dy_UBT = (self.y1_ubt - self.y0_ubt) / grid_ubt[1]


    def digitize_UBT(self, position: torch.tensor):
        """
        Digitizes the position for the UBT detector.
        :param position: Tensor of shape (N, 2) with x and y coordinates.
        :return: Tensor of digitized indices.
        """
        x = position[:, 0]
        y = position[:, 1]
        x_index = torch.floor((x - self.x0_ubt) / self.dx_UBT).long()
        y_index = torch.floor((y - self.y0_ubt) / self.dy_UBT).long()
        z_cell = self.z0 * torch.ones_like(x_index, dtype=torch.float32)  # z is constant for UBT
        y_cell = self.y0_ubt + y_index * self.dy_UBT
        x_cell = self.x0_ubt + x_index * self.dx_UBT
        return x_cell, y_cell, z_cell
    def digitize(self, position: torch.tensor, detector: int):
        """Digitizes the position for the SBT detector and times the operation.
        :param position: Tensor of shape (N, 3) with x, y, and z coordinates.
        :param detector: The detector ID (e.g., -2, -3 for horizontal, -4, -5 for vertical).
        :return: Tensor of digitized indices.
        """
        assert detector in [-1, -2, -3, -4, -5]
        start_time = time.time()
        if detector == -1:  # UBT
            result = self.digitize_UBT(position)
        elif detector in [-2, -3]:  # Horizontal SBT
            result = self.digitize_SBT_horizontal(position)
        elif detector in [-4, -5]:  # Vertical SBT
            result = self.digitize_SBT_vertical(position)
        else:
            raise ValueError("Invalid detector ID. Use -1 for UBT, -2 or -3 for horizontal SBT, -4 or -5 for vertical SBT.")
        elapsed = time.time() - start_time
        print(f"Digitization for detector {detector} took {elapsed:.6f} seconds.")
        return result
        
    def digitize_SBT_vertical(self, position: torch.tensor):
        """Digitizes the position for the vertical SBT detector.
        :param position: Tensor of shape (N, 3) with x, y, and z coordinates.
        :param detector: The detector ID (e.g., -4, -5).
        :return: Tensor of digitized indices.
        """
        length_hat = torch.sqrt(self.length**2 - (self.dX1 - self.dX2)**2)  # Adjusted length for vertical slab
        y0 = -self.dY2
        nz = self.grid_sbt[2]
        ny = self.grid_sbt[1]
        dy = (2*self.dY2) / ny
        dz = length_hat / nz
        print("dy:", dy)
        print("dz:", dz)

        z = position[:, 2]
        y = position[:, 1]
        x = position[:, 0]
        y_index = torch.floor((y - y0) / dy).long()
        z_index = torch.floor((z - self.z0) / dz).long()

        z_cell = self.z0 + z_index * dz
        y_cell = y0 + y_index * dy
        x_cell = z_cell * (self.dX2 - self.dX1) / self.length
        x_cell *= x.sign()  # Adjust x_cell sign based on x position
        return x_cell, y_cell, z_cell
    def digitize_SBT_horizontal(self, position: torch.tensor):
        """Digitizes the position for the horizontal SBT detector.
        :param position: Tensor of shape (N, 3) with x, y, and z coordinates.
        :return: Tensor of digitized indices.
        """
        length_hat = torch.sqrt(self.length**2 - (self.dY2 - self.dY1)**2)
        x0 = -self.dX2
        nz = self.grid_sbt[2]
        nx = self.grid_sbt[0]
        dx = (2*self.dX2) / nx
        dz = length_hat / nz
        print("dx:", dx)
        print("dz:", dz)

        z = position[:, 2]
        x = position[:, 0]
        y = position[:, 1]
        x_index = torch.floor((x - x0) / dx).long()
        z_index = torch.floor((z - self.z0) / dz).long()

        z_cell = self.z0 + z_index * dz
        x_cell = x0 + x_index * dx
        y_cell = z_cell * (self.dY2 - self.dY1) / self.length
        y_cell *= y.sign()  # Adjust y_cell sign based on y position
        return x_cell, y_cell, z_cell









if __name__ == "__main__":
    with h5py.File("/disk/users/lprate/share_data/full_sample_decay_vessel.h5", "r") as f:
        px = np.concatenate(f["px"][:])
        py = np.concatenate(f["py"][:])
        pz = np.concatenate(f["pz"][:])
        x = np.concatenate(f["x"][:])
        y = np.concatenate(f["y"][:])
        z = np.concatenate(f["z"][:])
        pdg_id = np.concatenate(f["pdg_id"][:])
        #detector = np.concatenate(f["detector"][:])
        #time = np.concatenate(f["time"][:])
    px = torch.tensor(px, dtype=torch.float32)
    py = torch.tensor(py, dtype=torch.float32)
    pz = torch.tensor(pz, dtype=torch.float32)
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    z = torch.tensor(z, dtype=torch.float32)
    pdg_id = torch.tensor(pdg_id, dtype=torch.int64)
    #detector = torch.tensor(detector, dtype=torch.int64)
    #time = torch.tensor(time, dtype=torch.float32)

    tagger = Tagger(grid_ubt=(50, 1), grid_sbt=(5, 8, 62))
    position = torch.stack((x, y, z), dim=1)
    x_cell, y_cell, z_cell = tagger.digitize(position, -1)  # UBT digitization
    print("UBT Digitized Cells:")
    print("x:", x_cell)
    print("y:", y_cell)
    print("z:", z_cell)
    
    # Example for horizontal SBT digitization
    x_cell_h, y_cell_h, z_cell_h = tagger.digitize(position, -2)
    print("\nHorizontal SBT Digitized Cells:")
    print("x:", x_cell_h)
    print("y:", y_cell_h)
    print("z:", z_cell_h)

    # Example for vertical SBT digitization
    x_cell_v, y_cell_v, z_cell_v = tagger.digitize(position, -4)
    print("\nVertical SBT Digitized Cells:")
    print("x:", x_cell_v)
    print("y:", y_cell_v)
    print("z:", z_cell_v)

    
