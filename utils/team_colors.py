import numpy as np
from sklearn.cluster import KMeans
import cv2
from pathlib import Path


class KitColorAnalyzer:

    def __init__(self, kit_colors):
        self.kit_colors = kit_colors
        self.kits_colors_lab = cv2.cvtColor(
            np.expand_dims(kit_colors.astype('float32')/ 255, 0),
            cv2.COLOR_RGB2Lab
        ).squeeze()

    def get_player_color(self, cropped, blur_kernel: int = 7):
        # convert colors and reshape
        pic = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        pic = cv2.medianBlur(pic, blur_kernel)
        pic = pic.reshape(-1, 3)

        # perform k-means clusterring and get labels
        pred = KMeans(n_clusters=2, random_state=0).fit(pic)
        labels = pred.labels_
        res = labels.reshape(cropped.shape[0], cropped.shape[1])

        # bbox border pixels - most of them must corresponds to background
        border = np.concatenate((
            res[:,0],
            res[:, -1],
            res[-1, 1:-1],
            res[0, 1:-1]
        ))
        bkg_cluster = np.argmax(np.bincount(border))
        player_claster = 1 - bkg_cluster
        player_color = pred.cluster_centers_[player_claster]
        return player_color

    def get_player_cluster_id(self, cropped_player):
        player_color = self.get_player_color(cropped_player)

        player_color = np.expand_dims(player_color, axis=(0, 1)).astype('float32')
        player_color_lab = cv2.cvtColor(player_color/ 255, cv2.COLOR_RGB2Lab).squeeze()
        idx = np.argmin(((self.kits_colors_lab - player_color_lab)**2).sum(axis=-1))
        return 0 if idx < 2 else 1
    
if __name__ == "__main__":
    p = Path("utils/cropped")
    frames = [cv2.imread(img.__str__()) for img in p.iterdir()]
    team_kits_colors = np.array([
        [182, 252, 164],
        [ 80,  92,  66],
        [221, 232, 250],
        [180, 143, 109],
    ])
    kca = KitColorAnalyzer(assumed_kit_colors=team_kits_colors)
