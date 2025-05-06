
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import cv2, yaml


class Custom_Dataset(Dataset):
  def __init__(self, targ_dir, split_txt):

    """Create a custom Dataset by subclassing torch.utils.data.Dataset

  Args:
    targ_dir: path to 01 / 02 …
    split_txt : path to train.txt  *or*  test.txt

    Returns:
      A dictionary of RGB, depth and their lables (R, t, bb) for each sample

  """

    self.targ_dir = Path(targ_dir)

    self.split_ids = []

    # Reading split ids
    with open(split_txt, "r") as f:
      for line in f:
        self.split_ids.append(line.strip())

    # Loading gt.yml
    with open(self.targ_dir / "gt.yml", "r") as f:
      gt_all = yaml.safe_load(f)


  def __len__(self):
    return len(self.split_ids)

  def __getitem(self, idx):
    name = self.split_ids[idx]
    int_convert = int(name)

    # Building the file paths for rgb and depth
    rgb_path = self.targ_dir / "rgb" / f"{name}.png"
    depth_path = self.targ_dir / "depth" / f"{name}.png"

    # Load RGB + depth
    rgb_img   = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
    depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

    # Load the ground truth pose
    pose_dict = self.gt_all[name][0]
    R = np.array(pose_dict["cam_R_m2c"], dtype=np.float32).reshape(3,3)
    t = np.array(pose_dict["cam_t_m2c"], dtype=np.float32)
    bb= np.array(pose_dict["obj_bb"], dtype=np.int64)

    sample = {
      "rgb"  : torch.from_numpy(rgb_img).permute(2,0,1).float() / 255.,
      "depth": torch.from_numpy(depth_img)[None].float(),
      "R"    : torch.from_numpy(R),
      "t"    : torch.from_numpy(t),
      "bb"   : torch.from_numpy(bb)
        }
    return sample
