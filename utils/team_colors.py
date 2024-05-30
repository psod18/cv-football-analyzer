from typing import List
import numpy as np
from sklearn.cluster import KMeans
import cv2
from pathlib import Path


class KitColorAnalyzer:

    def __init__(self, guessed_kit_colors: np.array):
        self.kit_colors = guessed_kit_colors

    def fit_colors(self, frames):
        print("Analize team kit's colors...")
        players_colors = []
        for box in frames:
            kit_color = self.get_player_color(box)
            players_colors.append(kit_color)

        self.kmeans = KMeans(n_clusters=4, init=self.kit_colors)
        self.kmeans.fit(players_colors)

        self.kit_colors = self.kmeans.cluster_centers_
        print("Kit colors map has been updated")

    def get_player_color(self, cropped):
        # convert colors and reshape
        pic = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        pic = pic.reshape(-1, 3)

        # perform k-means clusterring and get labels
        pred = KMeans(n_clusters=2, random_state=0).fit(pic)
        labels = pred.labels_
        res = labels.reshape(cropped.shape[0], cropped.shape[1])

        # get corner pixels - most of them must corresponds to background
        corners = [res[0,0], res[0, -1], res[-1, 0], res[-1, -1]]
        bkg_cluster = max(corners, key=corners.count)
        player_claster = 1 - bkg_cluster
        player_color = pred.cluster_centers_[player_claster]
        return player_color

    def get_player_team_lbl(self, cropped_player):
        player_color = self.get_player_color(cropped_player)
        kit_id = self.kmeans.predict(player_color.reshape(1,-1))
        team_id = 0 if kit_id < 2 else 1
        return team_id
    
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
    kca.fit_colors(frames=frames)
